# Graph/worker_sub_graph.py
from langgraph.graph import StateGraph, START, END
from Graph.rag_sub_graph import get_rag_subgraph
from Nodes.worker_agent import get_worker_agent_node
from Nodes.tool_nodes import get_tool_nodes
from Edges.route_worker_tools import route_worker_tools
from State.banking_state import WorkerState

def get_worker_subgraph(all_mcp_tools):
    """Compiles the worker and its tools into an isolated sub-graph."""
    
    # Initialize the nodes
    worker_agent_node, _ = get_worker_agent_node(all_mcp_tools)
    safe_tools_node, sensitive_tools_node = get_tool_nodes(all_mcp_tools)
    rag_subgraph = get_rag_subgraph()

    # 🌟 Build the mini-graph
    builder = StateGraph(WorkerState)
    builder.add_node("worker_agent", worker_agent_node)
    builder.add_node("safe_tools_node", safe_tools_node)
    builder.add_node("sensitive_tools_node", sensitive_tools_node)
    builder.add_node("rag_subgraph", rag_subgraph)

    # Start by running the agent
    builder.add_edge(START, "worker_agent")

    # Tools always loop back to the agent
    builder.add_edge("safe_tools_node", "worker_agent")
    builder.add_edge("sensitive_tools_node", "worker_agent")
    builder.add_edge("rag_subgraph", "worker_agent")

    # 🌟 Wrap the final output before returning to the parent graph
    def format_worker_output(state: WorkerState):
        """Extracts the final LLM message and pushes it to worker_responses."""
        final_message = state["messages"][-1].content
        return {"worker_responses": [f"Task {state['task'].task_id} Result: {final_message}"]}

    builder.add_node("format_output", format_worker_output)
    
    # 🌟 SINGLE CONDITIONAL EDGE: Agent routes to tools, or to the formatter
    builder.add_conditional_edges(
        "worker_agent",
        route_worker_tools,
        {
            "safe_tools_node": "safe_tools_node",
            "sensitive_tools_node": "sensitive_tools_node",
            "rag_subgraph": "rag_subgraph",
            END: "format_output" # Intercept END to format the output first
        }
    )
    
    # After formatting, the sub-graph officially ends and returns to the Orchestrator
    builder.add_edge("format_output", END)

    return builder.compile(interrupt_before=["sensitive_tools_node"])