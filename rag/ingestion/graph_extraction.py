"""
Entity/relationship extraction, writing into the NetworkX+SQLite store.
LLMGraphTransformer prompts an LLM to read raw text and emit
(entity)-[relationship]->(entity) triples -- e.g.
("NovaGrid Energy")-[ACQUIRED]->("SolarPeak Systems"). We flatten those
into plain (source, relation, target) tuples and hand them to the graph
store, which owns both the in-memory graph and its SQLite persistence.

Runs on the pre-contextual-prefix chunks (raw split text, no LLM-generated
context blurb prepended), NOT whole pages. This was originally "whole raw
pages" on the theory that entity extraction wants full sentences with
their original grammar -- true in principle, but a dense page can push
the extraction prompt (page text + LLMGraphTransformer's own fairly long
internal instructions) past a small local model's completion budget,
producing truncated JSON the parser can't read (openai.
LengthFinishReasonError). Chunk-boundary artifacts are a real but minor
cost against that failure mode. If extraction quality suffers noticeably
at chunk boundaries, revisit with a larger chunk size for extraction
specifically, or run on raw pages only for providers with a large enough
completion budget to handle it reliably (OpenAI's default is generous
enough; Gemma 4 via Ollama proved not to be, at least not without this
change).
"""
import logging
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer

from Config.llm_config import fast_llm
from Utils.logger import get_logger
from rag.retrieval.graph_store import get_graph_store

logger = get_logger(__name__)


def extract_and_store_graph(docs: List[Document], source_doc: str) -> Tuple[int, int]:
    logger.info("Extracting entities/relationships from %d document(s) for '%s'", len(docs), source_doc)
    # max_tokens=4096: extraction output is structured JSON listing every
    # node and relationship found -- a dense page can produce a lot of
    # entities, so this needs real headroom, not whatever a provider's
    # default completion budget happens to be.

    transformer = LLMGraphTransformer(llm=fast_llm)

    graph_documents = transformer.convert_to_graph_documents(docs)

    store = get_graph_store()
    total_nodes = 0
    total_relationships = 0

    for gd in graph_documents:
        triples = [
            (rel.source.id, rel.type, rel.target.id) for rel in gd.relationships
        ]
        store.add_triples(triples, source_doc=source_doc)
        total_nodes += len(gd.nodes)
        total_relationships += len(gd.relationships)

    logger.info(
        "Graph extraction complete for '%s': %d nodes, %d relationships",
        source_doc, total_nodes, total_relationships,
    )
    return total_nodes, total_relationships
