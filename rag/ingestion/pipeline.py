"""
The single orchestration point for ingestion: load+validate -> chunk ->
contextualize (full document, see chunking.py) -> embed+store (vector)
-> extract+store (graph). The /upload endpoint calls only this function
and knows nothing about chunking, the graph store, or the LLM provider
switch -- that's the point of modularity: api/upload.py depends on this
function's signature, not on any of the modules it calls internally.
"""
import time
from typing import Dict

from rag.ingestion.chunking import add_contextual_prefix, split_document
from rag.ingestion.graph_extraction import extract_and_store_graph
from rag.ingestion.loaders import load_document
from rag.retrieval.vector_store import get_vector_store
from Utils.logger import get_logger

logger = get_logger(__name__)


def ingest_file(file_path: str, original_filename: str) -> Dict:
    start = time.monotonic()
    logger.info("=== Ingestion started: '%s' ===", original_filename)

    # Raises ValueError if the document exceeds the configured page/word
    # cap -- caught and turned into a 400 response in api/upload.py.
    docs = load_document(file_path)

    for d in docs:
        d.metadata["source"] = original_filename

    full_text = "\n".join(d.page_content for d in docs)

    chunks = split_document(docs)
    contextual_chunks = add_contextual_prefix(chunks, full_text)

    vector_store = get_vector_store()
    vector_store.add_documents(contextual_chunks)
    logger.info("Stored %d contextualized chunks in the vector store", len(contextual_chunks))

    node_count, rel_count = extract_and_store_graph(chunks, source_doc=original_filename)

    elapsed = time.monotonic() - start
    logger.info(
        "=== Ingestion complete: '%s' in %.1fs -- %d chunks, %d entities, %d relationships ===",
        original_filename, elapsed, len(contextual_chunks), node_count, rel_count,
    )

    return {
        "chunks_created": len(contextual_chunks),
        "entities_extracted": node_count,
        "relationships_extracted": rel_count,
    }
