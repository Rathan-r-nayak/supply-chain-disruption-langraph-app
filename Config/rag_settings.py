"""
Centralized settings. Every module reads config from here instead of
os.environ directly -- this is what makes the provider switch
(llm_provider) meaningful: it's the one place chat model AND embedding
model selection both branch from.
"""
from typing import Optional

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Chroma (vector store) ---
    chroma_persist_dir: str = "./data/rag/chroma_db"
    chroma_collection_name: str = "documents"

    # --- Graph store: NetworkX (in-memory) + SQLite (persistence) ---
    graph_db_path: str = "./data/rag/graph_store.db"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Upload limits ---
    max_pdf_pages: int = 30
    max_txt_words: int = 15000

    # --- Logging ---
    log_level: str = "INFO"

settings = Settings()