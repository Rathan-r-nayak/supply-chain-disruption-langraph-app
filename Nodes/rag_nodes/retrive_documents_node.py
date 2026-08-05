# Nodes/rag_nodes/retrive_documents_node.py

from typing import List

from rag.retrieval.graph_store import graph_search
from rag.retrieval.vector_store import get_vector_store
from State.rag_state import HybridDocuments, RagState, VectorFact
from Utils.logger import get_logger

logger = get_logger("RETRIEVE_NODE")

def retrieve_node(state: RagState):
    logger.info("--- 📥 RETRIEVING HYBRID CONTEXT (VECTOR + GRAPH) ---")
    
    question = state.get("question", "").strip()
    if not question and state.get("messages"):
        question = state["messages"][-1].content.strip()
        
    logger.info(f"📥 Query for Hybrid Retrieval: '{question}'")

    # 1. Vector Search
    try:
        store = get_vector_store()
        hits = store.similarity_search(question, k=5)
        vector_chunks: List[VectorFact] = [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": float(score),
            }
            for doc, score in hits
        ]
        logger.info(f"🔍 Vector Search returned {len(vector_chunks)} hit(s).")
    except Exception as e:
        logger.error(f"❌ Vector Search failed: {e}")
        vector_chunks = []

    # 2. Graph Search
    try:
        graph_facts = graph_search(question)
        logger.info(f"🕸️ Graph Search returned {len(graph_facts)} fact(s).")
    except Exception as e:
        logger.error(f"❌ Graph Search failed: {e}")
        graph_facts = []

    # 3. Construct Hybrid Context
    document: HybridDocuments = {
        "vector_facts": vector_chunks,
        "graph_facts_used": graph_facts,
    }
    
    logger.info(f"✅ Hybrid context assembled: {len(vector_chunks)} vector chunk(s), {len(graph_facts)} graph fact(s).")

    return {"documents": document}