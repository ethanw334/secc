import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
#from langchain.chains import create_retrieval_chain, create_history_aware_retriever
#from langchain_core.messages import HumanMessage
import logging
from dotenv import load_dotenv
import chromadb
from chromadb.config import Settings

# Document Directory
document_dir = "PDFs/Smart_Home_Ex/SE_Artifacts"

# Select Models
LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

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
    Creates a more sophisticated RAG chain with query decomposition and multi-step reasoning.
    """
    logging.info("Creating the RAG chain...")
    llm = OllamaLLM(model=LLM_MODEL)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # retrieve top k most similar chunks

    # First, define the chain to generate sub-questions
    query_transform_prompt = ChatPromptTemplate.from_template("""
        You are an expert systems engineer. Your task is to analyze a complex question about systems engineering documents and break it down into a list of specific, actionable sub-questions.
        Each sub-question should be formulated to enable a semantic search against a vector database of systems engineering documents (requirements, architecture, etc.).
        
        Example:
        - Original Question: "What are the top inconsistencies of the system?"
        - Decomposed Questions: ["What are the key requirements?", "What is the system architecture?", "List the key subsystems and their functions.", "Identify any discrepancies between the requirements and the architecture."]
        
        Now, decompose the following question into a list of sub-questions. Respond only with the list, one item per line, without any other text.
        
        Original Question: {input}
    """)

    query_transformer = query_transform_prompt | llm # query_transform_prompt is passed as input to the llm

    # Second, define the chain to perform the final synthesis
    synthesis_prompt = ChatPromptTemplate.from_template("""
        You are a seasoned systems engineer performing an analysis of systems engineering documents. Your task is to identify inconsistencies, gaps, or compliance issues by comparing the provided contexts.
        
        Contexts:
        {context}
        
        Question:
        {input}
        
        Analyze the contexts and provide a detailed answer, citing specific examples and document sources where possible. If you find no inconsistencies, state that clearly.
    """)
    synthesis_chain = create_stuff_documents_chain(llm, synthesis_prompt)

    return query_transformer, synthesis_chain, retriever

def main():
    """
    Main function to run the multi-step RAG application.
    """
    load_dotenv()  # Load environment variables from a .env file if it exists

    documents = load_documents(document_dir)
    if not documents:
        return

    chunks = split_documents(documents)
    if not chunks:
        return

    vector_store = setup_vector_store(chunks)

    # Create the components of the advanced RAG chain
    query_transformer, synthesis_chain, retriever = create_rag_chain(vector_store)

    logging.info("RAG system is ready. Ask a question or type 'exit' to quit.")

    while True:
        question = input("\nYour question: ")
        if question.lower() == 'exit':
            break

        try:
            # Step 1: Decompose the complex query
            logging.info("Decomposing complex query...")
            sub_queries = query_transformer.invoke({"input": question}).strip().split('\n')
            logging.info(f"Generated sub-queries: {sub_queries}")
            
            all_retrieved_docs = []
            # Step 2: Retrieve documents for each sub-query
            for sub_q in sub_queries:
                retrieved_docs = retriever.invoke(sub_q)
                all_retrieved_docs.extend(retrieved_docs)
            
            # Remove duplicates based on page_content
            unique_docs = {doc.page_content: doc for doc in all_retrieved_docs}.values()
            
            # Step 3: Synthesize a final answer using all retrieved documents
            logging.info("Synthesizing final answer from all retrieved documents...")
            response = synthesis_chain.invoke({"context": unique_docs, "input": question})
            
            print("\nAnswer:", response)
            
            print("\n--- Retrieved Chunks ---")
            for i, doc in enumerate(unique_docs):
                source = doc.metadata.get('source', 'Unknown source')
                page = doc.metadata.get('page', 'N/A') + 1 if isinstance(doc.metadata.get('page'), int) else 'N/A'
                print(f"Chunk {i+1} from {source} (Page: {page}):\n{doc.page_content}\n")
            print("------------------------")

        except Exception as e:
            logging.error(f"An error occurred while processing the question: {e}")

if __name__ == "__main__":
    main()