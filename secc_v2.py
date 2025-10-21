import os
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from ollama import chat
import uuid 
import datetime 
import sqlite3
import hashlib
import sys
import gc
import re

# Select Model and PDF Directory
LLM = 'llama3.1:8b'
PDF_DIRECTORY = r'PDFs'

# --- Database Configuration ---
DATABASE_NAME = 'secc_artifacts.db'
METADATA_TABLE = 'artifact_log'

# Output Report
REPORT_FILENAME = "analysis_report.txt"

# ==============================================================================
# 1. DATA STRUCTURES (Pydantic Models and Enums)
# ==============================================================================

# Define the possible Category and Severity values
CategoryEnum = Literal["Syntax and Quality", "Traceability", "Semantic and Context", "Interface, Safety, and Cyber"]
SeverityEnum = Literal["Low", "Medium", "High", "Critical"]

# Define the structure for a single finding object
class Finding(BaseModel):
    FindingID: str = Field(description="Unique identifier for the finding (e.g., TRACE-001, SEM-002).")
    SourceArtifacts: List[str] = Field(description="List of artifacts (original filenames) involved in the issue.", min_length=1)
    FindingText: str = Field(description="A detailed description of the issue, conflict, or gap found during cross-comparison.")
    Category: CategoryEnum = Field(description="The classification of the issue, mapping to FR 3.3.")
    ConfidenceScore: float = Field(description="The system's confidence in this finding, from 0.0 (low) to 1.0 (high).", ge=0.0, le=1.0)
    SeverityLevel: SeverityEnum = Field(description="The estimated impact/severity of the finding.")

# Define the main structure (the array of findings)
class FindingsList(BaseModel):
    inconsistencies: List[Finding]

# Generate the JSON schema for the LLM
json_schema = FindingsList.model_json_schema()

# Define the structure for storing document metadata (Simulates Local DB)
class ArtifactMetadata(BaseModel):
    Filename: str = Field(description="Original uploaded filename.")
    VersionID: str = Field(description="Unique version identifier for this content state.")
    AuthorPlaceholder: str = Field(description="Placeholder for the document author/uploader.")
    UploadTimestamp: str = Field(description="Timestamp of when the document was ingested.")
    FileSizeBytes: int = Field(description="Size of the file in bytes.")
    ContentHash: str = Field(description="SHA256 hash of the PDF file content for change detection.")
    MarkdownContent: str = Field(description="The converted markdown content of the artifact.")

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

def initialize_database():
    """Initializes the SQLite database and creates the metadata table with new columns."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # SQL to create the table
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
                Filename TEXT NOT NULL,
                VersionID TEXT PRIMARY KEY,
                AuthorPlaceholder TEXT,
                UploadTimestamp TEXT,
                FileSizeBytes INTEGER,
                ContentHash TEXT UNIQUE,  
                MarkdownContent TEXT      
            );
        """)
        
        conn.commit()
        print(f"Database initialized: '{DATABASE_NAME}' with table '{METADATA_TABLE}'.")
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
    finally:
        if conn:
            conn.close()

def log_metadata_to_db(metadata: ArtifactMetadata):
    """Inserts a single ArtifactMetadata record into the SQLite database."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # SQL to insert data
        cursor.execute(f"""
            INSERT INTO {METADATA_TABLE} 
            (Filename, VersionID, AuthorPlaceholder, UploadTimestamp, FileSizeBytes, ContentHash, MarkdownContent) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.Filename, 
            metadata.VersionID, 
            metadata.AuthorPlaceholder, 
            metadata.UploadTimestamp, 
            metadata.FileSizeBytes,
            metadata.ContentHash,
            metadata.MarkdownContent
        ))
        
        conn.commit()
    except sqlite3.IntegrityError:
        # Catches duplicate ContentHash (meaning the file content hasn't changed)
        print(f"Warning: A record with this ContentHash/VersionID already exists. Skipping insertion.")
    except sqlite3.Error as e:
        print(f"Error logging metadata: {e}")
    finally:
        if conn:
            conn.close()

def check_for_cached_content(content_hash: str) -> Optional[str]:
    """Checks the database for an existing ContentHash and returns the Markdown content if found."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()
        
        # Query to retrieve the Markdown content based on the hash
        cursor.execute(f"SELECT MarkdownContent FROM {METADATA_TABLE} WHERE ContentHash = ?", (content_hash,))
        result = cursor.fetchone()
        
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Error checking cache: {e}")
        return None
    finally:
        if conn:
            conn.close()


# --- PDF Ingestion and Formatting (FR 3.1) ---
def get_pdfs_in_directory(directory_path, converter: PdfConverter):
    """
    Finds PDFs, calculates a content hash, uses the database cache to skip conversion 
    if content is unchanged, and converts/logs new content with traceability tags.

    Args:
        directory_path (str): The path to the directory containing PDFs.
        converter (PdfConverter): An initialized Marker PdfConverter object.
    
    Returns:
        dict: A dictionary mapping filename (str) to tagged Markdown content (str).
    """
    pdf_contents = {}
    
    # Check if the directory exists and is not empty
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return pdf_contents
        
    pdf_paths = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
    
    if not pdf_paths:
        print(f"Warning: No PDF files found in {directory_path}. Cannot proceed with analysis.")
        return pdf_contents

    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        print(f'Processing: {filename}')
        
        try:
            # 1. Read file bytes and Calculate Content Hash (SHA256)
            # This is fast and tells us if the file content has changed
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            
            # 2. Check Database for Cache
            # This function retrieves the full, tagged Markdown content if the hash exists.
            cached_content = check_for_cached_content(content_hash)
            
            if cached_content:
                # Content found and hasn't changed, skip expensive conversion
                print(f'Cached (Hash: {content_hash[:8]}). Using previous content.')
                text = cached_content
                
            else:
                # Content is new or updated.
                print(f'New or updated content. Converting (Slow step) and logging to database...')
                
                # Marker conversion logic (The expensive step)
                rendered = converter(pdf_path)
                text_unprocessed, _, _ = text_from_rendered(rendered) 
                
                # --- APPLY TRACEABILITY TAGS (Unique Tags for Every Section) ---
                text = tag_markdown_sections(text_unprocessed, filename) 
                
                # 3. Collect Metadata and Log New Content to DB
                file_stats = os.stat(pdf_path)
                version_id = str(uuid.uuid4())[:8] # Generate a new version ID for the new content state
                
                metadata = ArtifactMetadata(
                    Filename=filename,
                    VersionID=version_id,
                    AuthorPlaceholder="SECC_Uploader", 
                    UploadTimestamp=datetime.datetime.now().isoformat(),
                    FileSizeBytes=file_stats.st_size,
                    ContentHash=content_hash, # Log the hash of the new content
                    MarkdownContent=text     # Log the new tagged content
                )
                log_metadata_to_db(metadata)

            # Store the final tagged text content for the current LLM analysis run
            pdf_contents[filename] = text
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return pdf_contents


def print_findings_list(findings_list: FindingsList, output_file: str = None):
    """
    Generates the list of findings in a clean, human-readable format 
    and outputs it to the terminal or a specified file.
    """
    total_findings = len(findings_list.inconsistencies)
    
    # 1. Determine the output destination
    if output_file:
        print(f"--- Writing analysis report to: {output_file} ---")
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = sys.stdout # Default to terminal output
        
    def write_line(text=""):
        f.write(text + '\n')

    # 2. Replicate existing logic using write_line
    write_line("=" * 70)
    write_line(f"| ANALYSIS REPORT: Found {total_findings} Inconsistencies |")
    write_line("=" * 70)

    for i, finding in enumerate(findings_list.inconsistencies, 1):
        write_line(f"\n--- FINDING {i} of {total_findings} (ID: {finding.FindingID}) " + "-"*20)

        # Basic metadata
        write_line(f"Severity: \t\t{finding.SeverityLevel}")
        write_line(f"Category: \t\t{finding.Category}")
        write_line(f"Confidence: \t\t{finding.ConfidenceScore:.1f} / 1.0")
        write_line(f"Sources: \t\t{' & '.join(finding.SourceArtifacts)}")

        # Detailed text block
        write_line("\n[Detailed Description]")
        text_lines = finding.FindingText.split('\n')
        for line in text_lines:
            write_line(f"   > {line.strip()}")

        write_line("-" * 70)

    # 3. Close the file if it was opened
    if output_file:
        f.close()

def tag_markdown_sections(markdown_text: str, filename: str) -> str:
    """
    Scans Markdown text and injects a unique ID tag into every heading for traceability.
    
    The format will be [FILE_SECTION_ID], e.g., [CONOPS-S01-2].
    """
    lines = markdown_text.split('\n')
    tagged_lines = []
    
    # Use the first 3 letters of the filename for a unique prefix (e.g., CON)
    prefix = filename[:3].upper() 
    section_counter = 0

    # Regex to find any Markdown header (starts with 1-6 '#' symbols)
    header_pattern = re.compile(r'^(#+)\s*(.*)') 

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Found a header line
            section_counter += 1
            header_level = len(match.group(1)) # e.g., '##' is level 2
            
            # Generate a unique ID: [FILE-S(SectionNumber)]
            tag = f"[{prefix}-S{section_counter:03d}]"
            
            # Reconstruct the line: ### [TAG] Original Header Text
            tagged_line = f"{match.group(1)} {tag} {match.group(2).strip()}"
            tagged_lines.append(tagged_line)
        else:
            # Not a header, keep the line as is
            tagged_lines.append(line)

    return "\n".join(tagged_lines)

# ==============================================================================
# 3. MAIN EXECUTION LOGIC
# ==============================================================================

if __name__ == "__main__":
    # 3.0 Initialize the local database (FR: logs metadata into a local database)
    initialize_database()

    # Initialize PDF Converter (FR 3.1)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    
    # 3.1 Ingest and format documents (FR 3.1)
    # Pass the initialized converter into the function
    artifact_contents = get_pdfs_in_directory(PDF_DIRECTORY, converter)

    # --- VRAM Management: Manual Cleanup ---
    print("Attempting to release Marker models from VRAM...")

    # 1. Explicitly delete the converter object.
    del converter 

    # 2. Force the Python garbage collector to run.
    gc.collect() 

    # 3. Explicitly empty the CUDA cache (most effective VRAM cleanup)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        # If torch is unavailable or fails, rely on system/gc
        print(f"Warning: Could not explicitly empty CUDA cache. Error: {e}")

    print("Marker model cleanup complete.")
    # --- Marker VRAM is now cleared, making room for the LLM ---

    if not artifact_contents:
        print("Analysis terminated due to missing PDF content.")
        exit()

    # Format the contents into a single string for the LLM
    document_context = ""
    for filename, content in artifact_contents.items():
        # Tagging content with the filename is crucial for SourceArtifacts traceability
        document_context += f"--- START OF ARTIFACT: {filename} ---\n{content}\n--- END OF ARTIFACT: {filename} ---\n\n"

    # --- 4. LLM System and User Prompt Setup (FR 3.2, 3.3) ---
    system_instruction = f'''
You are an expert Systems Engineering (SE) Auditor. Your task is to perform a meticulous cross-comparison and analysis of the provided system artifacts.

**GOAL (FR 3.2):** Identify and classify all inconsistencies, conflicts, ambiguities, and gaps between the documents.

**CRITERIA FOR CLASSIFICATION (FR 3.3):**
1.  **Syntax and Quality:** Poor grammar, ambiguous language, passive voice, missing units, or non-atomic requirements.
2.  **Traceability:** A requirement/component in one document is not referenced, mapped, or addressed in another.
3.  **Semantic and Context:** Two artifacts describe the same system aspect but use conflicting or inconsistent values, terminology, or operational context.
4.  **Interface, Safety, and Cyber:** Missing or conflicting interface definitions, unaddressed safety hazards, or missing security/cyber-defense measures.

Ensure every finding includes a numeric ConfidenceScore (0.0-1.0) and a string SeverityLevel.
'''

    user_message = f'''
**ARTIFACTS FOR CROSS-COMPARISON**
---
{document_context}
---

**TASK:**
Perform the cross-comparison audit based on your system instructions. Analyze the entire set of artifacts for discrepancies.
---
'''
    # --- 5. LLM Call and Pydantic Validation ---
    print("--- Sending request to LLM (ollama chat) ---")
    response = chat(
        model=LLM,
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': user_message}
        ],
        format=json_schema, 
        options={'temperature': 0, # more deterministic output
                 'keep_alive': 0} # offloads model weights immediately after chat
    )

    # Validate and parse the JSON response into the Pydantic model
    inconsistencies = FindingsList.model_validate_json(response.message.content)
    
    # --- 6. Output Generation ---
    print_findings_list(inconsistencies, output_file=REPORT_FILENAME)