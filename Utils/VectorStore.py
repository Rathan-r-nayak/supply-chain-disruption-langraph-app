import os
from langchain_chroma import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.documents import Document
from Config.llm_config import embedding_model
from Utils.Logger import get_logger

logger = get_logger("VECTOR_STORE")

# 1. Define Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(PROJECT_ROOT, "DataStore", "chroma_db")

def get_vector_store():
    """Returns the persistent ChromaDB instance using Azure Embeddings."""
    return Chroma(
        collection_name="it_manuals",
        embedding_function= embedding_model,
        persist_directory=CHROMA_PATH
    )

def add_documents_to_store(chunks: list[Document]):
    """
    Takes a list of LangChain Documents and saves them permanently.
    """
    try:
        vector_store = get_vector_store()
        vector_store.add_documents(documents=chunks)
        logger.info(f"Successfully saved {len(chunks)} chunks to Vector DB at {CHROMA_PATH}")
        return True
    except Exception as e:
        logger.error(f"Vector DB Error: {e}")
        return False
    
# Add this to the bottom of Utils/VectorStore.py

def get_indexed_files() -> list:
    """
    Queries the Vector DB to retrieve a list of uniquely indexed filenames.
    """
    try:
        vector_store = get_vector_store()
        
        # In ChromaDB, we can fetch a specific subset of data.
        # Fetching a large number (e.g., 10000) ensures we get the metadata for most chunks.
        collection = vector_store._collection
        results = collection.get(include=["metadatas"])
        
        metadatas = results.get("metadatas", [])
        
        # Extract unique source names using a set
        unique_files = set()
        for meta in metadatas:
            if meta and "source" in meta:
                unique_files.add(meta["source"])
                
        # Return a sorted list
        return sorted(list(unique_files))
        
    except Exception as e:
        logger.error(f"Error fetching indexed files: {e}")
        return []