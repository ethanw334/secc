import os
from pydantic import BaseModel, Field
from typing import List, Literal
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from ollama import chat

# Select Model and PDF Directory
LLM = 'llama3.1:8b'
PDF_DIRECTORY = r'PDFs'

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

# ==============================================================================
# 2. HELPER FUNCTIONS
# ==============================================================================

# --- PDF Ingestion and Formatting (FR 3.1) ---
def get_pdfs_in_directory(directory_path, converter: PdfConverter):
    """
    Finds PDFs, converts them to text using 'marker', and returns a dictionary 
    mapping the filename to the content for traceability.
    
    Args:
        directory_path (str): The path to the directory containing PDFs.
        converter (PdfConverter): An initialized Marker PdfConverter object.
    
    Returns:
        dict: A dictionary mapping filename (str) to content (str).
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
        print(f'Loading and converting: {filename}')
        
        try:
            # Marker conversion logic
            rendered = converter(pdf_path)
            text, _, _ = text_from_rendered(rendered)
            
            # Store the content with the original filename as the key
            pdf_contents[filename] = text
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            
    return pdf_contents


def print_findings_list(findings_list: FindingsList):
    """
    Prints the list of findings in a clean, human-readable format.
    """
    total_findings = len(findings_list.inconsistencies)
    print("=" * 70)
    print(f"| ANALYSIS REPORT: Found {total_findings} Inconsistencies |")
    print("=" * 70)

    for i, finding in enumerate(findings_list.inconsistencies, 1):
        print(f"\n--- FINDING {i} of {total_findings} (ID: {finding.FindingID}) " + "-"*20)

        # Basic metadata
        print(f"Severity: \t\t{finding.SeverityLevel}")
        print(f"Category: \t\t{finding.Category}")
        print(f"Confidence: \t\t{finding.ConfidenceScore:.1f} / 1.0")
        print(f"Sources: \t\t{' & '.join(finding.SourceArtifacts)}")

        # Detailed text block
        print("\n[Detailed Description]")
        # Use an indented block for the finding text for better readability
        text_lines = finding.FindingText.split('\n')
        for line in text_lines:
            print(f"  > {line.strip()}")

        print("-" * 70)

# ==============================================================================
# 3. MAIN EXECUTION LOGIC
# ==============================================================================

if __name__ == "__main__":
    # Initialize PDF Converter (FR 3.1)
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )
    
    # 3.1 Ingest and format documents (FR 3.1)
    # Pass the initialized converter into the function
    artifact_contents = get_pdfs_in_directory(PDF_DIRECTORY, converter) 

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
        options={'temperature': 0}
    )

    # Validate and parse the JSON response into the Pydantic model
    inconsistencies = FindingsList.model_validate_json(response.message.content)
    
    # --- 6. Output Generation ---
    print_findings_list(inconsistencies)