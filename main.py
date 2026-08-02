from langgraph.graph import StateGraph, START, END

# --- State Imports ---
from Nodes.vision_node import vision_node
from Nodes.input_guardrail import input_guardrail_node
from Nodes.output_guardrail import output_guardrail_node
from Nodes.remember_node import remember_node
from Nodes.recall_node import recall_node
from State.banking_state import BankingState

# --- Node Imports ---
from Nodes.guardrail_node import guardrail_node
from Nodes.primary_classifier_node import triage_router
from Nodes.orchestrator import orchestrator_node
from Nodes.aggregator_node import aggregator_node
from Graph.worker_sub_graph import get_worker_subgraph

# --- Edge Imports ---
from Edges.guardrail_edge import guardrail_edge
from Edges.primary_classifier_route_edge import route_triage
from Edges.route_orchestration import orchestrator_router
import os
os.environ["LANGGRAPH_STRICT_MSGPACK"] = "false" # Prevents deserialization strict blocking for custom tasks

def build_graph(all_mcp_tools, checkpointer=None, ltm_store=None):
    """
    Assembles and compiles the map-reduce banking agent.
    Pass your PostgreSQL or MemorySaver checkpointer here.
    """
    # account_agent, account_tools, account_tools_condition = get_account_agent_nodes(all_mcp_tools=all_mcp_tools)
    # transaction_agent, safe_tools_node, sensitive_tools_node = get_transaction_agent_nodes(all_mcp_tools)
    # knowledge_agent, knowledge_tools_node, knowledge_tools_condition = get_knowledge_agent_nodes(all_mcp_tools)
    worker_subgraph = get_worker_subgraph(all_mcp_tools)

    workflow = StateGraph(BankingState)

    # 1. Add Nodes
    workflow.add_node("recall_node", recall_node)
    workflow.add_node("vision_node", vision_node)
    # workflow.add_node("guardrail_node", guardrail_node)
    workflow.add_node("input_guardrail", input_guardrail_node)
    workflow.add_node("output_guardrail", output_guardrail_node)
    workflow.add_node("triage_router", triage_router)
    workflow.add_node("orchestrator", orchestrator_node)
    # 🌟 The entire sub-graph acts as one node
    workflow.add_node("worker_subgraph", worker_subgraph) 
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("remember_node", remember_node)
    
    # workflow.add_node("retrieve_node", retrieve_node)
    # workflow.add_node("grade_docs_node", grade_docs_node)
    # workflow.add_node("rewrite_node", rewrite_node)
    # workflow.add_node("rag_format_response_node", rag_format_response_node)


    
    # 2. Add Edges & Conditional Routing
    workflow.add_edge(START, "recall_node")
    workflow.add_edge("recall_node", "vision_node")        # 🌟 Route to Vision
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
    workflow.add_edge("remember_node", END)

    app = workflow.compile(checkpointer=checkpointer, store=ltm_store)
    return app