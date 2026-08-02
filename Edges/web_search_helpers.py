from langchain_core.messages import AIMessage
from State.rag_state import RagState
from Utils.logger import get_logger

logger = get_logger("RAG_SUBGRAPH")

def ask_web_search(state: RagState):
    logger.info("--- 🛑 ASKING USER FOR WEB SEARCH (HITL) ---")
    # Append the question to the state. The UI will show this to the user.
    return {"messages": [AIMessage(content="I couldn't find this in our internal policies. Would you like me to search the web? (Yes/No)")]}


# --- 🌟 ROUTING LOGIC ---
def check_relevance(state: RagState):
    score = state.get("relevance_score", "yes")
    retries = state.get("knowledge_retries", 0)
    
    if score == "yes":
        return "generate_node"
    elif retries >= 2:
        logger.warning(f"⚠️ Max retries ({retries}) reached. Asking for Web Search.")
        return "ask_web_search"
    else:
        return "rewrite_node"

def check_web_search_approval(state: RagState):
    """Evaluates the user's Yes/No answer after the graph resumes."""
    last_msg = state.get("messages", [])[-1].content.lower()
    if "yes" in last_msg or "y" in last_msg:
        logger.info("✅ User approved web search.")
        return "web_search_node"
    
    logger.info("❌ User denied web search. Forcing generation with existing docs.")
    return "generate_node"

