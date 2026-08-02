"""
Two responsibilities:
1. split_document() -- pure mechanical splitting. No LLM calls.
2. add_contextual_prefix() -- Contextual Retrieval: for every chunk, ask
   the configured LLM (get_llm(), provider set by LLM_PROVIDER) to write a
   1-2 sentence blurb situating the chunk within the WHOLE document, and
   prepend it before embedding.

   Full document, not a truncated excerpt: this is viable specifically
   because uploads are capped at 20-30 pages / ~15,000 words (enforced in
   loaders.py) -- comfortably inside any model's context window. That cap
   is what makes this the right design here; a longer-document version of
   this project would need the hierarchical-summary approach discussed
   separately instead.
"""
import logging
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from Config.llm_config import fast_llm
from Config.settings import settings
from Utils.Logger import get_logger

logger = get_logger(__name__)

_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You situate a chunk of text within its parent document. "
            "Given the full document and one chunk from it, write a concise "
            "1-2 sentence context describing what this chunk is about and "
            "where it fits in the document, so the chunk is understandable "
            "on its own. Return ONLY the context sentences, nothing else.",
        ),
        ("human", "Full document:\n{document}\n\nChunk:\n{chunk}"),
    ]
)


def split_document(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split into %d chunks (chunk_size=%d, overlap=%d)", len(chunks), settings.chunk_size, settings.chunk_overlap)
    return chunks


def add_contextual_prefix(chunks: List[Document], full_text: str) -> List[Document]:
    logger.info("Generating contextual prefixes for %d chunks (full-document, ~%d chars)", len(chunks), len(full_text))
    chain = _CONTEXT_PROMPT | fast_llm

    enriched: List[Document] = []
    for i, chunk in enumerate(chunks, start=1):
        context = chain.invoke(
            {"document": full_text, "chunk": chunk.page_content}
        ).content.strip()

        enriched.append(
            Document(
                page_content=f"{context}\n\n{chunk.page_content}",
                metadata=chunk.metadata,
            )
        )
        if i % 10 == 0 or i == len(chunks):
            logger.info("Contextualized %d/%d chunks", i, len(chunks))
    return enriched
