import os
import time
import uuid
import shutil
import asyncio
import gc
import ollama
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import core_logic

# --- Configuration ---
UPLOAD_DIRECTORY = "temp_uploads"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# --- FastAPI App Initialization ---
app = FastAPI()

# Configure CORS to allow the React frontend (running on port 5173 by default)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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

@app.get("/models")
def get_available_models():
    """
    Fetches the list of locally downloaded models from Ollama
    and formats their size into GB.
    """
    try:
        response = ollama.list()
        
        models_data = []
        for m in response.models:
            # Convert bytes to Gigabytes (1000^3)
            size_gb = m.size / (1000 ** 3)
            
            models_data.append({
                "name": m.model,
                "size_label": f"{size_gb:.1f} GB"
            })
            
        return {"models": models_data}
        
    except Exception as e:
        print(f"Error fetching models: {e}")
        return {"models": []}
    
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
        # Receive the specific model name (e.g., "llama3.1:8b" or "gpt-5-mini")
        data = await websocket.receive_json()

        print(f"DEBUG: Received WebSocket Data: {data}")

        session_id = data.get("session_id")
        selected_model = data.get("model", core_logic.LLM) # Default to a safe local fallback

        start_time = time.time()
        
        if not session_id:
            await send_log("Error: No session_id provided.")
            return

        session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
        if not os.path.isdir(session_dir):
            await send_log(f"Error: Session directory not found: {session_dir}")
            return

        # Initialize PDF Converter
        await send_log("Initializing PDF converter (Marker)... This may take a moment.")
        converter = core_logic.PdfConverter(
            artifact_dict=core_logic.create_model_dict(),
        )

        # Ingest and format documents
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
                # Read file bytes and Calculate Hash
                with open(pdf_path, 'rb') as f:
                    pdf_bytes = f.read()
                content_hash = core_logic.hashlib.sha256(pdf_bytes).hexdigest()
                
                # Check Database for Cache
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
                    
                    # Collect Metadata and Log
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
        del converter 
        gc.collect() 
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except ImportError:
            pass # No torch

        if not artifact_contents:
            await send_log("Analysis terminated due to missing PDF content.")
            return
            
        # --- LLM Prompt Setup ---
        await send_log("Formatting document context for LLM...")
        document_context = ""
        for filename, content in artifact_contents.items():
            document_context += f"--- START OF ARTIFACT: {filename} ---\n{content}\n--- END OF ARTIFACT: {filename} ---\n\n"

        system_instruction = core_logic.system_instruction
        user_message = f'''
**ARTIFACTS FOR CROSS-COMPARISON**
---
{document_context}
---
**TASK:**
Perform the cross-comparison audit based on your system instructions. Analyze the entire set of artifacts for discrepancies.
---
'''
        # --- LLM Call ---
        await send_log(f"Sending request to {selected_model} ...")

        # Define the sync function that will run in a separate thread
        def run_llm_sync():
            if selected_model == core_logic.OPENAI_MODEL:
                if not core_logic.OPENAI_API_KEY:
                    raise ValueError("OpenAI API Key not found.")
                
                client = core_logic.OpenAI(api_key=core_logic.OPENAI_API_KEY)
                
                completion = client.beta.chat.completions.parse(
                    model=core_logic.OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=core_logic.FindingsList,
                    #temperature=0
                )
                
                # Extract usage from the raw completion object
                usage = completion.usage
                usage_stats = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "model": core_logic.OPENAI_MODEL
                }
                
                return completion.choices[0].message.parsed, usage_stats
                
            else:
                # Local (Ollama)
                response = core_logic.chat(
                    model=selected_model,
                    messages=[
                        {'role': 'system', 'content': system_instruction},
                        {'role': 'user', 'content': user_message}
                    ],
                    format=core_logic.json_schema, 
                    options={'temperature': 0, 'keep_alive': 0} 
                )
                
                return core_logic.FindingsList.model_validate_json(response.message.content), None

        # Run the logic
        try:
            inconsistencies, usage_stats = await asyncio.to_thread(run_llm_sync)
        except Exception as e:
            await send_log(f"LLM Error: {str(e)}")
            return

        await send_log("LLM response received. Calculating metrics...")
        
        # --- Calculate Cost ---
        total_cost = 0.0
        token_str = ""
        
        if usage_stats and selected_model == core_logic.OPENAI_MODEL:
            p_tokens = usage_stats["prompt_tokens"]
            c_tokens = usage_stats["completion_tokens"]
            
            # Formula: (num tokens / 1M) * Price
            cost_input = (p_tokens / 1_000_000) * core_logic.PRICE_GPT5_MINI_INPUT
            cost_output = (c_tokens / 1_000_000) * core_logic.PRICE_GPT5_MINI_OUTPUT
            total_cost = cost_input + cost_output
            
            token_str = f"({p_tokens + c_tokens} tokens)"
            
        else:
            total_cost = 0.0
            token_str = f"(Local: {selected_model})"

        # --- Health Report & Timing ---
        health_report = core_logic.calculate_health_report(inconsistencies)
        
        end_time = time.time()
        duration_seconds = end_time - start_time

        # time formatting
        if duration_seconds < 60:
            time_str = f"{duration_seconds:.1f}s"
        else:
            minutes = int(duration_seconds // 60)
            seconds = duration_seconds % 60
            time_str = f"{minutes}m {seconds:.1f}s"
            
        health_report["execution_time"] = time_str
        
        # Inject cost into report
        if total_cost == 0:
            health_report["cost"] = "Free"
        elif total_cost < 0.01:
            health_report["cost"] = f"${total_cost:.4f}"
        else:
            health_report["cost"] = f"${total_cost:.2f}"
            
        health_report["token_info"] = token_str

        # --- Send Result ---
        response_data = {
            "findings": inconsistencies.model_dump(),
            "health_report": health_report
        }
        
        await send_log(f"Analysis complete in {time_str}. Cost: {health_report['cost']}.")
        await websocket.send_json({"type": "result", "data": response_data})
        
        # --- Cleanup ---
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