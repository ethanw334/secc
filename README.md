Retrieval-Augmented Generation (RAG) System
1. Introduction
This project is a self-contained, offline-capable Retrieval-Augmented Generation (RAG) system built with Python. It allows users to ask questions about a custom knowledge base—composed of their own PDF documents—and receive accurate, contextually grounded answers.

The system leverages a stack of open-source technologies to ensure data privacy and operational independence from third-party APIs. It's designed to be modular and easy to set up on a local machine.

2. Features
Local LLM Integration: Uses the Ollama framework to run open-source models (like llama3.2) directly on your machine.

Document-Based Answering: Provides answers exclusively based on the content of your provided PDF files.

GPU Acceleration: Automatically leverages a compatible GPU (if available) via the Ollama server for significantly faster performance.

3. Prerequisites
Before you begin, ensure you have the following installed on your system:

Python 3.10 or a later version. (had issues with 3.13, but worked with 3.10)

Git for cloning the repository.

Ollama: Follow the instructions at ollama.ai to install the Ollama application. After installation, download the necessary models by running these commands in your terminal:

```
ollama pull llama3.2
ollama pull nomic-embed-text
```

A PDFs folder in the root directory of this project. Place all the PDF documents you want to use as your knowledge base inside this folder.

4. Installation
Follow these steps to set up the project:

Clone the repository:

```
git clone https: https://github.com/ethanw334/rag-system.git
cd your-repo-name
```
Create a virtual environment (recommended):

```
python -m venv venv
```

# On Windows
```
venv\Scripts\activate
```
# On macOS/Linux
```
source venv/bin/activate
```

Install Python dependencies:
```
pip install -r requirements.txt
```

5. Usage
To run the RAG application, simply execute the main Python script from your terminal:
```
python rag_app.py
```

The first time you run the script, it will create a new ChromaDB vector store named chroma_db in the project directory. This process can take a few minutes depending on the number of documents.

6. File Descriptions
rag_app.py: The main application script. This file contains all the logic for document ingestion, vectorization, and the RAG query-answering loop.

requirements.txt: A list of all Python libraries required for the project.

7. Contributing
Contributions are welcome! If you find a bug or have an idea for a new feature, please open an issue or submit a pull request.

