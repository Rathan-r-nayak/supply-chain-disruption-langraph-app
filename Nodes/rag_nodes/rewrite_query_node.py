from langchain_classic.prompts import ChatPromptTemplate
from Config.llm_config import fast_llm
from State.rag_state import RagState
from Utils.Logger import get_logger
# Import your local DB connections (e.g., Neo4j, Milvus/FAISS/Chroma)
# from DB.vector_store import get_vector_store
# from DB.graph_store import get_graph_store

logger = get_logger("REWRITE_QUERY_NODE")

def rewrite_node(state: RagState):
    logger.info("--- ✍️ REWRITING KNOWLEDGE QUERY ---")
    question = state.get("question", "")
    retries = state.get("knowledge_retries", 0)

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at optimizing search queries for hybrid vector/graph databases. Output a refined, highly specific search query as plain text."),
        ("human", "Original question: {question}")
    ])

    new_query = str((rewrite_prompt | fast_llm).invoke({"question": question}).content)
    logger.info(f"New Query: {new_query}")

    # Update the question for the next retrieval loop
    return {
        "question": new_query,
        "knowledge_retries": retries + 1
    }