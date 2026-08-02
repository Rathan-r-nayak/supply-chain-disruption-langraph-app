"""
BaseVectorStore is the seam that makes this modular: every other module
talks to get_vector_store() and the BaseVectorStore interface only. To
swap Chroma for Qdrant/FAISS/pgvector later, write ONE new class
implementing the same two methods and change ONE line in
get_vector_store() -- nothing in ingestion/, agents/, or api/ changes.

Embeddings follow the same LLM_PROVIDER switch as the chat model:
"gemma" -> nomic-embed-text via native Ollama (no OpenAI dependency at
all), "openai" -> OpenAIEmbeddings. This is what makes a fully local,
zero-API-cost POC possible.

IMPORTANT -- embedding spaces are not interchangeable: vectors produced
by nomic-embed-text and vectors produced by OpenAI's embedding model are
NOT comparable. If you switch LLM_PROVIDER after already ingesting
documents, existing vectors in chroma_persist_dir were built in a
different embedding space -- similarity search against them silently
returns meaningless results, no error. Switching providers requires
deleting chroma_persist_dir and re-ingesting everything, not just
changing the env var.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import httpx
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from Config.llm_config import embedding_model
from Config.settings import settings
from Utils.Logger import get_logger

logger = get_logger("VECTORE_STORE")

client = httpx.Client(verify=False)


def _get_embeddings() -> Embeddings:
    logger.info(
        "Using embedding model: %s (%s)",
        embedding_model.model,
        embedding_model.openai_api_base
    )
    return embedding_model


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, docs: List[Document]) -> None: ...

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]: ...


class ChromaVectorStore(BaseVectorStore):
    def __init__(self) -> None:
        self._embeddings = _get_embeddings()
        self._store = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embeddings,
            persist_directory=settings.chroma_persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )

    def add_documents(self, docs: List[Document]) -> None:
        logger.info(
            "Adding %d documents to Chroma collection '%s'",
            len(docs),
            settings.chroma_collection_name,
        )
        self._store.add_documents(docs)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        logger.info(
            "Vector search for %r returned %d result(s)", query[:80], len(results)
        )
        return results


_vector_store_singleton: BaseVectorStore | None = None


def get_vector_store() -> BaseVectorStore:
    global _vector_store_singleton
    if _vector_store_singleton is None:
        _vector_store_singleton = ChromaVectorStore()
    return _vector_store_singleton
