# Systems Engineering Command Center (SECC)

## 1. Introduction

This Systems Engineering Command Center is a offline-capable, Python-based application that processes user-uploaded PDF technical artifacts and automatically performs cross-document consistency checks. It generates a final report summarizing all identified context issues and inconsistencies across the document set.

## 2. Features
Local LLM Integration: Uses the Ollama framework to run open-source models (like llama3.2) directly on your machine.

GPU Acceleration: Automatically leverages a compatible GPU (if available) via the Ollama server for significantly faster performance.

## 3. Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.10+ and PyTorch

Ollama - Follow the instructions at ollama.ai to install the Ollama application. After installation, download the necessary models by running these commands in your terminal:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

Place all the PDF documents you want to use as your knowledge base inside the PDFs folder. 

## 4. Installation
Follow these steps to set up the project:

Clone the repository:

```
git clone https: https://github.com/ethanw334/secc.git
cd secc
```
Create a virtual environment (recommended):

```
python -m venv venv
```

### On Windows
```
venv\Scripts\activate
```
### On macOS/Linux
```
source venv/bin/activate
```

Install Python dependencies:
```
pip install -r requirements.txt
```

## 5. Usage
To run the SECC application, simply execute the main Python script from your terminal:
```
python se_artifact.py
```

## 6. File Descriptions

se_artifact.py: The SECC application script using full context.

rag_app.py: The RAG application script. This file contains all the logic for document ingestion, vectorization, and the RAG query-answering loop.
