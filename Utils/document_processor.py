import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from Utils.vector_store import add_documents_to_store
from Utils.logger import get_logger

logger = get_logger("DOCUMENT_PROCESSOR")

def process_and_index_files(file_info_list: list, category: str, item: str):
    """
    Processes files and indexes them into ChromaDB.
    Expected input: file_info_list = [(temp_path, original_filename), ...]
    """
    if not file_info_list:
        return False
        
    all_chunks = []
    
    # 1. Unpack the list of tuples
    for path, original_name in file_info_list:
        if not os.path.exists(path):
            logger.warning(f"File not found: {path}")
            continue
            
        try:
            # 2. Load the file
            loader = PyPDFLoader(path) if path.lower().endswith(".pdf") else TextLoader(path)
            documents = loader.load()
            
            # 3. Chunk it
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            
            # 4. Inject Metadata (Aligned with Ticket schema)
            for chunk in chunks:
                chunk.metadata["catalog_category"] = category
                chunk.metadata["catalog_item"] = item
                chunk.metadata["source"] = original_name
            
            all_chunks.extend(chunks)
            logger.info(f"Processed {len(chunks)} chunks from {original_name}")
            
        except Exception as e:
            logger.error(f"Error processing {original_name}: {e}")
            continue

    # 5. Save to Vector Store
    if all_chunks:
        success = add_documents_to_store(all_chunks)
        
        # 6. Cleanup files only after successful indexing
        if success:
            for path, _ in file_info_list:
                try: 
                    if os.path.exists(path):
                        os.remove(path)
                except Exception as e:
                    logger.warning(f"Cleanup failed for {path}: {e}")
        return success
        
    return False