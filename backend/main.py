import os
import uuid
import shutil
import asyncio
import gc
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# Import all models and functions from your original script
import core_logic

# --- Configuration ---
UPLOAD_DIRECTORY = "temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- FastAPI App Initialization ---
app = FastAPI()

# Configure CORS to allow the React frontend (running on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Update if your React port is different
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    """Initialize the database when the server starts."""
    print("Initializing database...")
    core_logic.initialize_database()
    print("Database initialized.")

# --- HTTP Endpoint: File Upload ---
@app.post("/upload")
async def create_upload_session(files: List[UploadFile] = File(...)):
    """
    Handles uploading multiple PDF files.
    It creates a unique session, saves the files, and returns the session_id.
    """
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
    return {"session_id": session_id}

def calculate_health_report(findings_list: core_logic.FindingsList) -> dict:
    """
    Calculates a Multi-Attribute Utility Theory (MAUT) risk-based health score from the list of findings.
    """
    # weights for each category (must sum to 1)
    weights = {
        "Syntax": 0.25,
        "Traceability": 0.25,
        "Semantic": 0.25,
        "Cybersecurity": 0.25 
    }
    
    # maximum acceptable/expected number of findings for category R (each must be > 0)
    max_findings = {
        "Syntax": 10,
        "Traceability": 10,
        "Semantic": 10,
        "Cybersecurity": 10
    }

    finding_counts = {
        "Syntax": 0,
        "Traceability": 0,
        "Semantic": 0,
        "Cybersecurity": 0
    }

    severity_counts = {
        "Low": 0,
        "Medium": 0,
        "High": 0,
        "Critical": 0
    }

    for finding in findings_list.inconsistencies:
        severity_counts[finding.SeverityLevel] += 1
        finding_counts[finding.Category] += 1

    r_category = {category: finding_counts[category] / max_findings[category] for category in max_findings}

    h = sum(weights[category] * r_category[category] for category in weights)

    # 4. Convert risk score to a 0-100 health score
    # We use a simple subtraction, capping at 0.
    health_score = max((1 - h) * 100, 0)

    # --- 5. Determine state message and level based on score ---
    state_message = ""
    state_level = ""
    
    if health_score == 100:
        state_message = "Excellent. No issues found."
        state_level = "pass"
    elif health_score >= 85:
        state_message = "Healthy. Only minor issues found."
        state_level = "pass"
    elif health_score >= 70:
        state_message = "Needs Review. Moderate issues detected."
        state_level = "review"
    else:
        state_message = "Critical Issues. System health is low."
        state_level = "danger"

    # --- 6. NEW: Add new fields to the return dictionary ---
    return {
        "score": round(health_score, 1),
        "total_findings": len(findings_list.inconsistencies),
        "critical_count": severity_counts["Critical"],
        "high_count": severity_counts["High"],
        "medium_count": severity_counts["Medium"],
        "low_count": severity_counts["Low"],
        "state_message": state_message,  
        "state_level": state_level       
    }

# --- WebSocket Endpoint: Analysis ---
@app.websocket("/ws/analysis")
async def websocket_endpoint(websocket: WebSocket):
    """
    Handles the long-running analysis over a WebSocket connection.
    """
    await websocket.accept()
    
    # Helper function to send logs back to the client
    async def send_log(message: str):
        print(message)  # Log to server console
        await websocket.send_json({"type": "log", "message": message})

    try:
        # 1. Wait for the client to send the session_id
        data = await websocket.receive_json()
        session_id = data.get("session_id")
        
        if not session_id:
            await send_log("Error: No session_id provided.")
            return

        session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
        if not os.path.isdir(session_dir):
            await send_log(f"Error: Session directory not found: {session_dir}")
            return

        # --- This is the logic from your script's main block, ---
        # --- adapted for the async WebSocket flow. ---

        # 3.0 Initialize PDF Converter
        await send_log("Initializing PDF converter (Marker)... This may take a moment.")
        converter = core_logic.PdfConverter(
            artifact_dict=core_logic.create_model_dict(),
        )

        # 3.1 Ingest and format documents
        # We must refactor the get_pdfs_in_directory to accept our log function
        
        artifact_contents = {}
        pdf_paths = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_paths:
            await send_log("Error: No PDF files found in session.")
            return
            
        await send_log(f"Found {len(pdf_paths)} PDF(s) to process.")

        for pdf_path in pdf_paths:
            filename = os.path.basename(pdf_path)
            await send_log(f'Processing: {filename}')
            
            try:
                # 1. Read file bytes and Calculate Hash
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                content_hash = core_logic.hashlib.sha256(pdf_bytes).hexdigest()
                
                # 2. Check Database for Cache
                cached_content = core_logic.check_for_cached_content(content_hash)
                
                if cached_content:
                    await send_log(f'Cached (Hash: {content_hash[:8]}). Using previous content.')
                    text = cached_content
                else:
                    await send_log(f'New or updated content. Converting (Slow step)...')
                    
                    # Run the sync, CPU/GPU-bound Marker conversion in a separate thread
                    def convert_pdf_sync():
                        rendered = converter(pdf_path) 
                        return core_logic.text_from_rendered(rendered)[0]

                    text = await asyncio.to_thread(convert_pdf_sync)
                    
                    await send_log(f'Conversion complete. Logging to database...')
                    
                    # 3. Collect Metadata and Log
                    file_stats = os.stat(pdf_path)
                    version_id = str(core_logic.uuid.uuid4())[:8] 
                    
                    metadata = core_logic.ArtifactMetadata(
                        Filename=filename,
                        VersionID=version_id,
                        AuthorPlaceholder="SECC_Uploader", 
                        UploadTimestamp=core_logic.datetime.datetime.now().isoformat(),
                        FileSizeBytes=file_stats.st_size,
                        ContentHash=content_hash, 
                        MarkdownContent=text
                    )
                    core_logic.log_metadata_to_db(metadata)

                artifact_contents[filename] = text
            
            except Exception as e:
                await send_log(f"Error processing {filename}: {e}")

        # --- VRAM Management ---
        await send_log("Releasing PDF converter from VRAM...")
        del converter 
        gc.collect() 
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            await send_log("VRAM cache cleared.")
        except ImportError:
            pass # No torch

        if not artifact_contents:
            await send_log("Analysis terminated due to missing PDF content.")
            return
            
        # --- 4. LLM Prompt Setup ---
        await send_log("Formatting document context for LLM...")
        document_context = ""
        for filename, content in artifact_contents.items():
            document_context += f"--- START OF ARTIFACT: {filename} ---\n{content}\n--- END OF ARTIFACT: {filename} ---\n\n"

        system_instruction = core_logic.system_instruction # Get from module
        user_message = f'''
**ARTIFACTS FOR CROSS-COMPARISON**
---
{document_context}
---
**TASK:**
Perform the cross-comparison audit based on your system instructions. Analyze the entire set of artifacts for discrepancies.
---
'''
        # --- 5. LLM Call ---
        await send_log(f"Sending request to LLM ({core_logic.LLM})... This is the final, slow step.")
        
        # The 'ollama.chat' function is synchronous, so we run it in a thread
        # to avoid blocking the async server.
        def run_ollama_sync():
            return core_logic.chat(
                model=core_logic.LLM,
                messages=[
                    {'role': 'system', 'content': system_instruction},
                    {'role': 'user', 'content': user_message}
                ],
                format=core_logic.json_schema, 
                options={'temperature': 0, 'keep_alive': 0} 
            )

        response = await asyncio.to_thread(run_ollama_sync)
        
        # --- THIS IS THE NEW CODE ---
        await send_log("LLM response received. Validating...")
        inconsistencies = core_logic.FindingsList.model_validate_json(response.message.content)

        await send_log("Calculating health score...")
        health_report = calculate_health_report(inconsistencies)

        # --- 6. Send Result ---
        # We now send a combined object with both findings and the report
        response_data = {
            "findings": inconsistencies.model_dump(),
            "health_report": health_report
        }

        await send_log("Analysis complete. Sending results.")
        await websocket.send_json({"type": "result", "data": response_data})
        
        # --- 7. Cleanup ---
        await send_log(f"Cleaning up session {session_id}.")
        shutil.rmtree(session_dir)
        await send_log("Session cleanup complete.")

    except WebSocketDisconnect:
        print(f"Client disconnected.")
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        try:
            await send_log(error_message) # Try to send error to client
        except:
            pass # Client might be gone
    finally:
        # Ensure session dir is cleaned up if an error occurred mid-process
        if 'session_dir' in locals() and os.path.isdir(session_dir):
            shutil.rmtree(session_dir)
            print(f"Cleaned up {session_dir} after error/disconnect.")
        await websocket.close()