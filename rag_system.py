# -----------------------------------------------------------------------------
# RAG System with LangChain and ChromaDB using Ollama
# -----------------------------------------------------------------------------
# This script builds a simple RAG (Retrieval-Augmented Generation) system.
# It reads documents from a specified 'PDFs' directory, splits them into
# chunks, creates embeddings, and stores them in a ChromaDB vector store.
# Finally, it uses a retrieval chain to answer questions based on the
# loaded documents, using an Ollama-hosted LLM.
#
# Prerequisites:
# 1. Install required libraries:
#    pip install -r requirements.txt
#
# 2. Install and run Ollama:
#    Follow instructions at https://ollama.ai/ to install Ollama.
#    Run the following commands in your terminal to download the necessary models:
#    ollama pull llama3.2
#    ollama pull nomic-embed-text
#
# 3. Place your PDF files inside the 'PDFs' directory.
# -----------------------------------------------------------------------------

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import logging
from dotenv import load_dotenv

# Select Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

# Import the ChromaDB client and settings
import chromadb
from chromadb.config import Settings

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents(directory):
    """
    Loads all PDF documents from a specified directory.
    """
    try:
        logging.info(f"Loading documents from directory: {directory}")
        if not os.path.exists(directory):
            logging.error(f"Directory not found: {directory}")
            return None
        loader = PyPDFDirectoryLoader(directory)
        documents = loader.load()
        if not documents:
            logging.warning("No documents found in the specified directory.")
        else:
            logging.info(f"Successfully loaded {len(documents)} document(s).")
        return documents
    except Exception as e:
        logging.error(f"An error occurred while loading documents: {e}")
        return None

def split_documents(documents):
    """
    Splits documents into smaller, manageable chunks.
    """
    logging.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Created {len(chunks)} chunks.")
    return chunks

def setup_vector_store(chunks, persist_directory="chroma_db"):
    """
    Creates and persists a ChromaDB vector store from document chunks.
    """
    logging.info("Creating embeddings and setting up vector store...")
    # Create a persistent ChromaDB client with telemetry disabled
    client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)
    )

    # Using Ollama embeddings
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    # Pass the client to the Chroma.from_documents method
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client
    )
    logging.info("Vector store created and persisted successfully.")
    return vector_store

def create_rag_chain(vector_store):
    """
    Creates the retrieval-augmented generation chain.
    """
    logging.info("Creating the RAG chain...")
    # Define the Ollama language model. Ensure 'llama2' is pulled via `ollama pull llama2`.
    llm = OllamaLLM(model=LLM_MODEL)

    # Define the prompt template for the RAG system
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context.
    If you do not know the answer, just say that you do not know, do not try to make up an answer.

    Context:
    {context}

    Question:
    {input}
    """)

    # Create the document combination chain
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Create the retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Create the full retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    logging.info("RAG chain created.")
    return retrieval_chain

def main():
    """
    Main function to run the RAG application.
    """
    load_dotenv()  # Load environment variables from a .env file if it exists

    # Load and process documents
    documents = load_documents("PDFs")
    if not documents:
        return

    chunks = split_documents(documents)
    if not chunks:
        return

    # Setup the vector store
    vector_store = setup_vector_store(chunks)

    # Create the RAG chain
    rag_chain = create_rag_chain(vector_store)

    logging.info("RAG system is ready. Ask a question or type 'exit' to quit.")

    # Main user interaction loop
    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break

        try:
            # Invoke the RAG chain with the user's question
            response = rag_chain.invoke({"input": question})
            # The 'answer' key contains the generated response
            print("\nAnswer:", response['answer'])
        except Exception as e:
            logging.error(f"An error occurred while processing the question: {e}")

if __name__ == "__main__":
    main()




