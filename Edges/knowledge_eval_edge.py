from State.banking_state import BankingState
from Utils.Logger import get_logger


logger = get_logger("KNOWLEDGE_EVAL_EDGE")


def check_relevance(state: BankingState):
    score = state.get("relevance_score", "yes")
    retries = state.get("knowledge_retries", 0)
    logger.info(f"🔍 CHECKING RELEVANCE EDGE: Score = '{score}', Retries = {retries}")
    
    if score == "yes":
        logger.info("✅ Relevant documents found. Routing back to Agent for synthesis.")
        return "knowledge_agent"
    elif retries >= 2:
        logger.warning(f"⚠️ Max retries ({retries}) reached. Forcing generation with current context.")
        return "knowledge_agent"
    else:
        logger.info(f"❌ Irrelevant documents (Attempt {retries + 1}). Routing to query rewriter.")
        return "rewrite_node"