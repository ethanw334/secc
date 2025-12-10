# Systems Engineering Command Center (SECC)

## 1. Introduction

This Systems Engineering Command Center is a offline-capable, Python-based application that processes user-uploaded PDF technical artifacts and automatically performs cross-document consistency checks. It generates a final report summarizing all identified context issues and inconsistencies across the document set.

## 2. Features
- Local LLM Integration: Uses the Ollama framework to run open-source models (like qwen3:8b) directly on your machine.

- GPU Acceleration: Automatically leverages a compatible GPU (if available) via the Ollama server for significantly faster performance.

## 3. Prerequisites
Before you begin, ensure you have the following installed on your system:

- Python 3.10+
- Node.js - https://nodejs.org/en/download
- Ollama - Follow the instructions at https://ollama.com/ to install the Ollama application. After installation, download the necessary models by running these commands in your terminal:

```
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

## 4. Installation
Follow these steps to set up the project:

1. Clone the repository:

```
git clone https: https://github.com/ethanw334/secc.git
cd secc
```
2. Create a virtual environment (recommended):

```
python -m venv venv
```

#### On Windows
```
venv\Scripts\activate
```
#### On macOS/Linux
```
source venv/bin/activate
```
- Alternatively with Conda:
```
conda create --name secc_env python=3.13
conda activate secc_env
```
3. Install PyTorch - https://pytorch.org/
4. Install Python dependencies:
```
pip install -r requirements.txt
```

5. Run the following:
```
cd frontend
npm install
```

6. To use OpenAI's GPT-5 mini:
- Add a .env file to the root directory
- Add the following line to the .env file: ```OPENAI_API_KEY = "your api key here"```

## 5. Usage

1. Start the backend:
```
cd backend
uvicorn main:app --reload
```
and keep this terminal open.

2. Open a new terminal and navigate to frontend folder:
```
cd frontend
npm run dev
```
Open the link http://localhost:5173/ in browser to use app.

Upload all PDF documents to the SECC and click Analyze to generate a report. 