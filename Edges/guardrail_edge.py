from langgraph.graph import END
from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger

logger = get_logger("GUARDRAIL_EDGE")

def guardrail_edge(state: SupplyChainState):
    """Routes to semantic cache if safe, otherwise terminates the graph."""
    if state.get("is_safe") is False:
        logger.warning("🛑 ROUTING: Guardrail failed (is_safe=False). Routing to END.")
        return END
    
    logger.info("➡️ ROUTING: Input is safe. Routing to 'check_semantic_cache'.")
    return "check_semantic_cache"