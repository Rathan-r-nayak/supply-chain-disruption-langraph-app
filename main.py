import os
from langgraph.graph import StateGraph, START, END

# --- State Imports ---
from Nodes.summarize_stm_node import summarize_conversation_node
from State.supply_chain_state import SupplyChainState

# --- Node Imports ---
from Nodes.vision_node import vision_node
from Nodes.input_guardrail import input_guardrail_node
from Nodes.output_guardrail import output_guardrail_node
from Nodes.remember_node import remember_node
from Nodes.recall_node import recall_node
from Nodes.primary_classifier_node import triage_router
from Nodes.orchestrator import orchestrator_node
from Nodes.aggregator_node import aggregator_node
from Graph.worker_sub_graph import get_worker_subgraph

# --- Edge Imports ---
from Edges.guardrail_edge import guardrail_edge
from Edges.primary_classifier_route_edge import route_triage
from Edges.route_orchestration import orchestrator_router

os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false"

def build_graph(all_mcp_tools, checkpointer=None, ltm_store=None):
    """
    Assembles and compiles the map-reduce supply chain agent.
    """
    worker_subgraph = get_worker_subgraph(all_mcp_tools)

    workflow = StateGraph(SupplyChainState)

    # 1. Add Nodes
    workflow.add_node("recall_node", recall_node)
    workflow.add_node("vision_node", vision_node)
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("triage_router", triage_router)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("worker_subgraph", worker_subgraph) 
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("remember_node", remember_node)
    workflow.add_node("summarize_conversation_node", summarize_conversation_node)
    
    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "recall_node")
    workflow.add_edge("recall_node", "vision_node")
    workflow.add_edge("vision_node", "input_guardrail")
    workflow.add_edge("recall_node", "input_guardrail")

    workflow.add_conditional_edges(
        "input_guardrail",
        guardrail_edge,
        {
            "triage_router": "triage_router",
            END: END
        }
    )
    
    workflow.add_conditional_edges(
        "triage_router",
        route_triage,
        {
            "orchestrator": "orchestrator",
            "remember_node": "remember_node"
        }
    )

    workflow.add_conditional_edges(
        "orchestrator", 
        orchestrator_router, 
        {
            "worker_subgraph": "worker_subgraph",
            "aggregator": "aggregator"
        }
    )

    workflow.add_edge("worker_subgraph", "orchestrator")
    workflow.add_edge("aggregator", "output_guardrail")
    workflow.add_edge("output_guardrail", "remember_node")
    workflow.add_edge("remember_node", "summarize_conversation_node")
    workflow.add_edge("summarize_conversation_node", END)

    app = workflow.compile(checkpointer=checkpointer, store=ltm_store)
    return app