import os
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
from dotenv import load_dotenv
from openai import OpenAI
import math

# --- GLOBAL CONFIGURATION ---
LLM = 'qwen3:8b' 
EMBEDDING_MODEL = 'nomic-embed-text'
PDF_DIRECTORY = r'PDFs'

# --- OPENAI CONFIGURATION ---
OPENAI_MODEL = 'gpt-5-mini'
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Pricing per million tokens
PRICE_GPT5_MINI_INPUT = 0.25
PRICE_GPT5_MINI_OUTPUT = 2.00

# --- Database & ChromaDB Configuration ---
DATABASE_NAME = 'secc_artifacts.db'
METADATA_TABLE = 'artifact_log'
CHROMA_COLLECTION = 'secc_artifact_collection'

# Output Report
REPORT_FILENAME = "analysis_report.txt"

# --- LLM System Prompt ---
system_instruction = f'''
You are a **Master Systems Engineering (SE) Auditor**. Your SOLE task is to perform a **FORENSIC CROSS-DOCUMENT COMPARISON** of the provided artifacts. Identify **ONLY** meaningful conflicts, gaps, and inconsistencies.

# **CRITICAL MANDATE:**
# DO NOT summarize requirements. Every finding MUST provide proof by **quoting the relevant text from the involved documents**, and state the filename of the source document for the quote.

**CATEGORY DEFINITIONS (STRICT):**
Classify every finding into exactly one of these four categories:
* **"Syntax":** Quality issues within a requirement (e.g., vague words like "fast", "user-friendly"; missing "shall" statements; untestable requirements).
* **"Traceability":** Orphans or gaps (e.g., a User Need with no downstream System Requirement; a Requirement with no associated Test Case).
* **"Semantic":** Factual contradictions between documents (e.g., Doc A says "12V battery", Doc B says "24V mains").
* **"Cybersecurity":** Missing security controls, conflicting authentication/encryption standards, or unaddressed threat vectors.

**MANDATORY FIELD FORMATS:**
* **FindingID:** You MUST generate a unique ID for each finding, using a prefix for the category. (e.g., `SEM-001`, `TRACE-001`, `CYBER-001`, `SYN-001`).
* **FindingText:** MUST use this format: "[ISSUE TYPE]: [Detailed description with QUOTES and FILENAMES]."
* **SourceArtifacts:** MUST be a list of EXACT filenames involved (e.g., ["srs.pdf", "sdd.pdf"]).
* **Category:** MUST be one of: "Syntax", "Traceability", "Semantic", "Cybersecurity".
* **ConfidenceScore:** MUST be a float between 0.0 and 1.0 (e.g., 0.95). **NO PERCENTAGES.**
* **SeverityLevel:** MUST be one of: "Low", "Medium", "High", "Critical".

If no inconsistencies are found, return an empty list.
'''

# ==============================================================================
# 1. DATA STRUCTURES (Pydantic Models and Enums)
# ==============================================================================

# Define the possible Category and Severity values
CategoryEnum = Literal["Syntax", "Traceability", "Semantic", "Cybersecurity"]
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

# Define the structure for storing document metadata
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

def calculate_health_report(findings_list: FindingsList) -> dict:
    """
    Calculates a Multi-Attribute Utility Theory (MAUT) risk-based health score from the list of findings.
    """
    # weights for each severity (must sum to 1)
    weights = {
        "Critical": 0.55,
        "High": 0.25,
        "Medium": 0.15,
        "Low": 0.05 
    }

    R_base = {
        "Critical": 0.60,
        "High": 0.45,
        "Medium": 0.30,
        "Low": 0.15 
    }

    k = {severity: - math.log(1 - R_base[severity]) for severity in R_base}

    # This dictionary will store the counts for the final report
    severity_counts = {
        "Critical": 0,
        "High": 0,
        "Medium": 0,
        "Low": 0
    }

    for finding in findings_list.inconsistencies:
        severity_counts[finding.SeverityLevel] += 1

    # Use severity_counts to calculate the ratio r
    r = {severity: 1 - math.exp(-k[severity] * severity_counts[severity]) for severity in weights}

    h = sum(weights[severity] * r[severity] for severity in weights)

    # Convert risk score to a 0-100 health score
    health_score = max((1 - h) * 100, 0)

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

# --- PDF Ingestion and Formatting ---
def get_pdfs_in_directory(directory_path, converter: PdfConverter):
    """
    Finds PDFs, caches the content, and converts/logs new content.
    The content is NOT tagged with section IDs anymore.
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
            # Read file bytes and Calculate Content Hash (SHA256)
            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()
            
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            
            # Check Database for Cache
            cached_content = check_for_cached_content(content_hash)
            
            if cached_content:
                print(f'Cached (Hash: {content_hash[:8]}). Using previous content.')
                text = cached_content
                
            else:
                print(f'New or updated content. Converting (Slow step) and logging to database...')
                
                # Marker conversion logic (The expensive step)
                rendered = converter(pdf_path) 
                text, _, _ = text_from_rendered(rendered) 
                                
                # Collect Metadata and Log New Content to DB
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

# --- Vectorization and RAG Preparation ---
def create_vector_store(artifact_contents: dict, collection_name: str, embedding_model_name: str):
    """
    Chunks Markdown content, embeds the chunks using Ollama, and stores 
    them in a local ChromaDB collection.
    """
    print("\n--- Starting Vectorization and ChromaDB Storage ---")
    
    # Initialize Ollama Embeddings (loads the model to VRAM)
    try:
        embeddings = OllamaEmbeddings(model=embedding_model_name)
    except Exception as e:
        print(f"Error initializing Ollama Embeddings: {e}")
        print(f"Please ensure the embedding model '{embedding_model_name}' is pulled ('ollama pull {embedding_model_name}')")
        return None

    # Initialize ChromaDB Client
    persist_directory = "chroma_db"
    chroma_client = chromadb.PersistentClient(path=persist_directory)
    
    collection = chroma_client.get_or_create_collection(name=collection_name)
    
    # Delete previous data
    try:
        print(f"Clearing existing documents from collection: {collection_name}")
        collection.delete(where={})
    except Exception:
        pass # Ignore if collection is empty or new
        
    # Process Each Artifact
    all_chunks: List[LangChainDocument] = []
    
    # Define headers to split by
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
            # page/chunk ID
            source_tag = f"{doc_type}-P{i+1:03d}" 
            
            # Build final, rich metadata dictionary
            chunk.metadata.update({
                "source_file": filename,
                "doc_type": doc_type,
                "source_tag": source_tag, # The unique section ID for retrieval
                "chunk_index": i
            })
            
            all_chunks.append(chunk)

    print(f"Total documents processed: {len(artifact_contents)}. Total chunks created: {len(all_chunks)}")
    
    # Embed and Store
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
        print(f"\n--- Writing analysis report to: {output_file} ---")
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
        write_line(f"Involved Artifacts: \t{' & '.join(finding.SourceArtifacts)}")

        write_line("\n[Detailed Description]")
        text_lines = finding.FindingText.split('\n')
        for line in text_lines:
            write_line(f"  > {line.strip()}")

        write_line("-" * 70)

    if output_file:
        f.close()