import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Load environment variables (API Key)
load_dotenv()

DB_PATH = "./db"

def process_documents(uploaded_files):
    """
    Takes uploaded PDF files, converts them to text chunks,
    and saves them to the Vector Database using HuggingFace embeddings.
    """
    documents = []
    
    if not uploaded_files:
        return False

    try:
        for file in uploaded_files:
            # Save uploaded file momentarily to a temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            # Load PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            documents.extend(docs)
            
            # Cleanup temp file
            os.remove(temp_file_path)

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(documents)

        # Create Embeddings (Local, Free, CPU-friendly)
        # We use 'all-MiniLM-L6-v2' which is fast and good for RAG
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Save to ChromaDB
        Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=DB_PATH
        )
        
        return True
        
    except Exception as e:
        print(f"Error in ingestion: {e}")
        return False