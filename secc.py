import os
import re
import datetime
import uuid
import hashlib
import sqlite3
import gc 
import sys 
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from ollama import chat
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document as LangChainDocument
import chromadb

# --- GLOBAL CONFIGURATION ---
LLM = 'llama3.1:8b'
EMBEDDING_MODEL = 'nomic-embed-text'
PDF_DIRECTORY = r'PDFs'

# --- Database & ChromaDB Configuration ---
DATABASE_NAME = 'secc_artifacts.db'
METADATA_TABLE = 'artifact_log'
CHROMA_COLLECTION = 'secc_artifact_collection'

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

# --- Database Management ---
def initialize_database():
    """Initializes the SQLite database and creates the metadata table with new columns."""
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_NAME)
        cursor = conn.cursor()

        # SQL to create the table, relying on ContentHash for uniqueness
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
        
        # Use INSERT OR IGNORE based on the ContentHash UNIQUE constraint
        cursor.execute(f"""
            INSERT OR IGNORE INTO {METADATA_TABLE} 
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
        
        if cursor.rowcount == 0:
             print(f"Metadata for {metadata.Filename} (Hash: {metadata.ContentHash[:8]}) already exists. Log ignored.")
        else:
             print(f"Logged new version V-ID {metadata.VersionID} for {metadata.Filename}.")
             
        conn.commit()
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
        
        cursor.execute(f"SELECT MarkdownContent FROM {METADATA_TABLE} WHERE ContentHash = ?", (content_hash,))
        result = cursor.fetchone()
        
        return result[0] if result else None
    except sqlite3.Error as e:
        print(f"Error checking cache: {e}")
        return None
    finally:
        if conn:
            conn.close()

# --- Markdown Processing ---
def tag_markdown_sections(markdown_text: str, filename: str) -> str:
    """
    Scans Markdown text and injects a unique ID tag into every heading for traceability.
    Format: [FILE_SECTION_ID], e.g., [CON-S001].
    """
    lines = markdown_text.split('\n')
    tagged_lines = []
    
    # Use the first 3 letters of the filename for a unique prefix (e.g., CON)
    prefix = filename.split('.')[0][:3].upper() 
    section_counter = 0

    # Regex to find any Markdown header (starts with 1-6 '#' symbols)
    header_pattern = re.compile(r'^(#+)\s*(.*)') 

    for line in lines:
        match = header_pattern.match(line)
        if match:
            # Found a header line
            section_counter += 1
            
            # Generate a unique ID: [FILE-S(SectionNumber)]
            tag = f"[{prefix}-S{section_counter:03d}]"
            
            # Reconstruct the line: ### [TAG] Original Header Text
            tagged_line = f"{match.group(1)} {tag} {match.group(2).strip()}"
            tagged_lines.append(tagged_line)
        else:
            # Not a header, keep the line as is
            tagged_lines.append(line)

    return "\n".join(tagged_lines)

# --- PDF Ingestion and Formatting (FR 3.1) ---
def get_pdfs_in_directory(directory_path, converter: PdfConverter):
    """
    Finds PDFs, caches the content, and converts/logs new content with traceability tags.
    """
    pdf_contents = {}
    
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
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            
            # 2. Check Database for Cache
            cached_content = check_for_cached_content(content_hash)
            
            if cached_content:
                print(f'Cached (Hash: {content_hash[:8]}). Using previous content.')
                text = cached_content
                
            else:
                print(f'New or updated content. Converting (Slow step) and logging to database...')
                
                # Marker conversion logic (The expensive step)
                rendered = converter(pdf_path)
                text_unprocessed, _, _ = text_from_rendered(rendered) 
                
                # --- APPLY TRACEABILITY TAGS ---
                text = tag_markdown_sections(text_unprocessed, filename) 
                
                # 3. Collect Metadata and Log New Content to DB
                file_stats = os.stat(pdf_path)
                version_id = str(uuid.uuid4())[:8] 
                
                metadata = ArtifactMetadata(
                    Filename=filename,
                    VersionID=version_id,
                    AuthorPlaceholder="SECC_Uploader", 
                    UploadTimestamp=datetime.datetime.now().isoformat(),
                    FileSizeBytes=file_stats.st_size,
                    ContentHash=content_hash, 
                    MarkdownContent=text     
                )
                log_metadata_to_db(metadata)

            pdf_contents[filename] = text
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    return pdf_contents

# --- Vectorization and RAG Prep (FR 3.4) ---
def create_vector_store(artifact_contents: dict, collection_name: str, embedding_model_name: str):
    """
    Chunks Markdown content, embeds the chunks using Ollama, and stores 
    them in a local ChromaDB collection.
    """
    print("\n--- Starting Vectorization and ChromaDB Storage ---")
    
    # 1. Initialize Ollama Embeddings (This loads the model to VRAM)
    try:
        embeddings = OllamaEmbeddings(model=embedding_model_name)
    except Exception as e:
        print(f"Error initializing Ollama Embeddings: {e}")
        print(f"Please ensure the embedding model '{embedding_model_name}' is pulled ('ollama pull {embedding_model_name}')")
        return None

    # 2. Initialize ChromaDB Client
    persist_directory = "chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Delete previous data
    try:
        print(f"Clearing existing documents from collection: {collection_name}")
        collection.delete(where={})
    except Exception:
        pass # Ignore if collection is empty or new
        

    # 3. Process Each Artifact
    all_chunks: List[LangChainDocument] = []
    
    # Define headers to split by. This relies on the tags we injected earlier.
    headers_to_split_on = [
        ("#", "SectionTitle"),
        ("##", "SectionTitle"),
        ("###", "SectionTitle"),
        ("####", "SectionTitle"),
    ]

    for filename, markdown_content in artifact_contents.items():
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False
        )
        
        chunks = markdown_splitter.split_text(markdown_content)
        
        doc_type = filename.split('.')[0].upper()
        
        for i, chunk in enumerate(chunks):
            # Extract the actual traceability tag (e.g., [CON-S001])
            tag_match = re.search(r'\[[A-Z]{3}-S\d{3}\]', chunk.page_content)
            source_tag = tag_match.group(0).strip('[]') if tag_match else f"{doc_type}-P{i+1:03d}"
            
            # Build final, rich metadata dictionary
            chunk.metadata.update({
                "source_file": filename,
                "doc_type": doc_type,
                "source_tag": source_tag, # The unique section ID for retrieval
                "chunk_index": i
            })
            
            all_chunks.append(chunk)

    print(f"Total documents processed: {len(artifact_contents)}. Total chunks created: {len(all_chunks)}")
    
    # 4. Embed and Store
    if all_chunks:
        print("Embedding and storing documents in ChromaDB... (Using OllamaEmbeddings)")
        
        try:
            # Use LangChain's helper to create the store, handling embedding
            Chroma.from_documents(
                documents=all_chunks,
                embedding=embeddings,
                collection_name=collection_name,
                persist_directory=persist_directory
            )
            print(f"Successfully stored {len(all_chunks)} embeddings in {collection_name}.")
            
        except Exception as e:
            print(f"Failed during embedding or storage: {e}")
    
    print("Vector store successfully created and ready for RAG.")

# --- Output Management ---
def print_findings_list(findings_list: FindingsList, output_file: str = None):
    """
    Generates the list of findings in a clean, human-readable format 
    and outputs it to the terminal or a specified file.
    """
    total_findings = len(findings_list.inconsistencies)
    
    if output_file:
        print(f"--- Writing analysis report to: {output_file} ---")
        f = open(output_file, 'w', encoding='utf-8')
    else:
        f = sys.stdout 
        
    def write_line(text=""):
        f.write(text + '\n')

    write_line("=" * 70)
    write_line(f"| ANALYSIS REPORT: Found {total_findings} Inconsistencies |")
    write_line("=" * 70)

    for i, finding in enumerate(findings_list.inconsistencies, 1):
        write_line(f"\n--- FINDING {i} of {total_findings} (ID: {finding.FindingID}) " + "-"*20)

        write_line(f"Severity: \t\t{finding.SeverityLevel}")
        write_line(f"Category: \t\t{finding.Category}")
        write_line(f"Confidence: \t\t{finding.ConfidenceScore:.1f} / 1.0")
        write_line(f"Sources: \t\t{' & '.join(finding.SourceArtifacts)}")

        write_line("\n[Detailed Description]")
        text_lines = finding.FindingText.split('\n')
        for line in text_lines:
            write_line(f"   > {line.strip()}")

        write_line("-" * 70)

    if output_file:
        f.close()

# ==============================================================================
# 3. MAIN EXECUTION LOGIC
# ==============================================================================

if __name__ == "__main__":
    
    # 3.0 Initialize the local database
    initialize_database()
    
    # Initialize PDF Converter (FR 3.1) - Loads models to VRAM
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    
    # 3.1 Ingest and format documents
    artifact_contents = get_pdfs_in_directory(PDF_DIRECTORY, converter) 

    # --- VRAM Management: Manual Cleanup (Marker) ---
    print("\nAttempting to release Marker models from VRAM...")
    del converter 
    gc.collect() 
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except:
        pass
    print("Marker model cleanup complete.")
    
    if not artifact_contents:
        print("Analysis terminated due to missing PDF content.")
        exit()

    # --- 3.2 Create Vector Store (FR 3.4) ---
    # This step loads the *embedding* model, uses it, and then explicitly unloads it. For later RAG implementation
    #create_vector_store(artifact_contents, CHROMA_COLLECTION, EMBEDDING_MODEL)
    
    # Format the contents into a single string for the final LLM cross-comparison
    document_context = ""
    for filename, content in artifact_contents.items():
        document_context += f"--- START OF ARTIFACT: {filename} ---\n{content}\n--- END OF ARTIFACT: {filename} ---\n\n"

    # --- 4. LLM System and User Prompt Setup (FR 3.2, 3.3) ---
    system_instruction = f'''
You are a **Master Systems Engineering (SE) Auditor and Forensic Analyst**. Your task is a **MANDATORY CROSS-DOCUMENT COMPARISON**. Identify **ONLY** conflicts, gaps, and inconsistencies between the provided artifacts.

**CRITICAL MANDATE:**
DO NOT summarize requirements. Every finding MUST provide proof by **quoting the relevant text from the involved documents**, including their full traceability tags.

**MANDATORY CITATION FORMAT (FindingText):**
Your description MUST be structured as: **[ISSUE TYPE]:** [Description of conflict].
Example Conflict: "SEMANTIC CONFLICT: The system power specification in **[SRS-S012]** states '12V DC', but the Hardware Design in **[SDD-H005]** specifies '24V AC'."
Example Traceability Gap: "TRACEABILITY GAP: The User Need for **'real-time alerts'** in **[CON-S005]** has **NO corresponding functional requirement or test case** found in the SRS or V&V documents."

**OBJECTIVES: Focus on Consistency and Traceability:**
1.  **Consistency Checks (Semantic/Context):** Detect all conflicting values, terminology, or operational constraints between documents.
2.  **Traceability Checks:** Verify that every requirement is fully defined, flows down, and is covered by verification/test plans. Identify any orphans or missing links.

**INSTRUCTIONS FOR OUTPUT GENERATION (STRICT):**
* **SourceEntities:** MUST list the specific section tags (e.g., [CON-S005], [SRS-S012], NFR-2.1) that are directly involved in the conflict or gap.
* **Category Mapping:** Use one of the four categories only.

**Note:** If the LLM doesn't have the specific tag (e.g., NFR-2.1), it MUST attempt to find the section tag (e.g., [SRS-S015]) for that requirement.
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
    print("\n--- Sending request to LLM (ollama chat) ---")
    response = chat(
        model=LLM,
        messages=[
            {'role': 'system', 'content': system_instruction},
            {'role': 'user', 'content': user_message}
        ],
        format=json_schema, 
        options={'temperature': 0, 
                 'keep_alive': 0 # Instructs Ollama to offload the model immediately after this request
                 } 
    )

    # Validate and parse the JSON response into the Pydantic model
    inconsistencies = FindingsList.model_validate_json(response.message.content)
    
    # --- 6. Output Generation ---
    print_findings_list(inconsistencies, output_file=REPORT_FILENAME)