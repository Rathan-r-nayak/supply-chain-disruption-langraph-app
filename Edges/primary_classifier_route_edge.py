from langgraph.graph import END
from State.supply_chain_state import SupplyChainState
from Utils.Logger import get_logger

logger = get_logger("Primary Classifier edge")

def route_triage(state: SupplyChainState):
    """
    Reads the state output from the triage_router.
    Routes to the orchestrator if a workflow is needed, otherwise ends the graph.
    """

    if(state.get("requires_workflow")):
        logger.info("➡️ ROUTING: Sending to Orchestrator for task planning.")
        return "orchestrator"

    logger.info("🛑 ROUTING: Direct response generated. Ending graph.")
    return "remember_node"