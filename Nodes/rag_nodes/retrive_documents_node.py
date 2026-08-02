# Nodes/rag_nodes/retrive_documents_node.py

from typing import List

from rag.retrieval.graph_store import graph_search
from rag.retrieval.vector_store import get_vector_store
from State.rag_state import HybridDocuments, RagState, VectorFact
from Utils.Logger import get_logger

logger = get_logger("RETRIEVE_NODE")

def retrieve_node(state: RagState):
    logger.info("--- 📥 RETRIEVING HYBRID CONTEXT (VECTOR + GRAPH) ---")
    
    question = state.get("question", "").strip()
    if not question and state.get("messages"):
        question = state["messages"][-1].content.strip()
        
    logger.info(f"user query: {question}")

    # 1. Vector Search
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

    # 2. Graph Search
    graph_facts = graph_search(question)

    # 3. Construct Hybrid Context
    document: HybridDocuments = {
        "vector_facts": vector_chunks,
        "graph_facts_used": graph_facts,
    }
    
    logger.info(f"✅ Hybrid context retrieved: {len(vector_chunks)} vector chunks, {len(graph_facts)} graph facts.")

    return {"documents": document}