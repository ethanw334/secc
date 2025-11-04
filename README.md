# Systems Engineering Command Center (SECC)

## 1. Introduction

This Systems Engineering Command Center is a offline-capable, Python-based application that processes user-uploaded PDF technical artifacts and automatically performs cross-document consistency checks. It generates a final report summarizing all identified context issues and inconsistencies across the document set.

## 2. Features
- Local LLM Integration: Uses the Ollama framework to run open-source models (like llama3.1:8b) directly on your machine.

- GPU Acceleration: Automatically leverages a compatible GPU (if available) via the Ollama server for significantly faster performance.

## 3. Prerequisites
Before you begin, ensure you have the following installed on your system:

- Python 3.10+
- PyTorch
- Node.js
- Ollama - Follow the instructions at https://ollama.com/ to install the Ollama application. After installation, download the necessary models by running these commands in your terminal:

```
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## 4. Installation
Follow these steps to set up the project:

- Clone the repository:

```
git clone https: https://github.com/ethanw334/secc.git
cd secc
```
- Create a virtual environment (recommended):

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
- Alternatively with Conda:
```
conda create --name rag_env python=3.13
conda activate rag_env
```
- Install Python dependencies:
```
pip install -r requirements.txt
```

## 5. Usage

Place all the PDF documents you want to use as your knowledge base inside the PDFs folder. 

To run the SECC application, simply execute the main Python script from your terminal:
```
python secc.py
```

The report will be output as analysis_report.txt

## 6. File Descriptions

secc.py: The SECC application script using all PDF data.

rag_system.py: The RAG application script. This file contains all the logic for document ingestion, vectorization, and the RAG query-answering loop.


## How to run with the frontend

1. Start the backend:
```
cd backend
uvicorn main:app --reload
```
and keep this terminal open.

2. Open a new terminal and navigate to frontend folder.
```
cd frontend
npm run dev
```
Open the link http://localhost:5173/ in browser to use app.