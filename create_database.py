import os
import shutil
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

def create_database(file_path: str, db_dir: str = "chroma_db") -> int:
    """
    Loads a PDF file, splits it into chunks, creates a new Chroma database,
    and returns the number of chunks created. Replaces any existing database.
    """
    # Load environment variables
    load_dotenv()
    
    # 1. Clean up existing database directory to start fresh
    if os.path.exists(db_dir):
        try:
            shutil.rmtree(db_dir)
        except Exception:
            # Fallback if Windows file lock is being stubborn
            import gc
            import time
            gc.collect()
            time.sleep(0.5)
            try:
                shutil.rmtree(db_dir)
            except Exception as e:
                raise RuntimeError(f"Could not clear existing database directory: {str(e)}")
            
    # 2. Load PDF document
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # 3. Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    if not chunks:
        raise ValueError("No text content could be extracted from the PDF.")
    
    # 4. Initialize embedding model
    embedding_model = MistralAIEmbeddings(model="mistral-embed")
    
    # 5. Create new Chroma database
    # We use an explicit PersistentClient to ensure we can close it and release file locks
    client = chromadb.PersistentClient(path=db_dir)
    try:
        vectorstore = Chroma.from_documents(
            client=client,
            documents=chunks,
            embedding=embedding_model
        )
    finally:
        # Guarantee client is closed to release file locks on Windows
        client.close()
        
    return len(chunks)