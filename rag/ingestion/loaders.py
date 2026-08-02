"""
Loader selection by extension, plus the size validation that makes
full-document contextual retrieval viable. Everything downstream
(chunking, embedding, graph extraction) only ever deals with LangChain's
Document type, never the raw file format -- adding a new format later
only touches this file.
"""
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

from Config.settings import settings
from Utils.Logger import get_logger

logger = get_logger(__name__)
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def load_document(file_path: str) -> List[Document]:
    ext = Path(file_path).suffix.lower()
    logger.info("Loading document: %s (type=%s)", file_path, ext)

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in (".txt", ".md", ".markdown"):
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    docs = loader.load()
    _validate_document_size(docs, ext)
    logger.info("Loaded '%s': %d document(s)/page(s)", file_path, len(docs))
    return docs


def _validate_document_size(docs: List[Document], ext: str) -> None:
    """
    Enforces the upload caps the architecture depends on: keeping a whole
    document small enough to fit in a model's context window in one shot,
    so chunking.py can send the FULL document with every chunk's context
    prompt instead of truncating or summarizing it.

    PyPDFLoader returns one Document per page, so len(docs) is the page
    count directly. Text/markdown has no "page" concept, so it's capped
    by word count instead.
    """
    if ext == ".pdf":
        page_count = len(docs)
        if page_count > settings.max_pdf_pages:
            logger.warning("Rejecting PDF: %d pages exceeds cap of %d", page_count, settings.max_pdf_pages)
            raise ValueError(
                f"PDF has {page_count} pages; the limit is {settings.max_pdf_pages} "
                f"pages (full-document contextual retrieval assumes small documents)."
            )
    else:
        word_count = sum(len(d.page_content.split()) for d in docs)
        if word_count > settings.max_txt_words:
            logger.warning("Rejecting document: ~%d words exceeds cap of %d", word_count, settings.max_txt_words)
            raise ValueError(
                f"Document has ~{word_count} words; the limit is {settings.max_txt_words} words."
            )
