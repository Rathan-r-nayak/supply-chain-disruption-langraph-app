from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig # 🌟 NEW IMPORT

from Graph.rag_sub_graph import get_rag_subgraph
from Nodes.worker_agent import get_worker_agent_node
from Nodes.tool_nodes import get_tool_nodes
from Edges.route_worker_tools import route_worker_tools
from State.supply_chain_state import WorkerState

from Utils.logger import get_logger

logger = get_logger("WORKER_SUBGRAPH")

def get_worker_subgraph(all_mcp_tools):
    """Compiles the worker and its tools into an isolated sub-graph."""
    
    worker_agent_node, _ = get_worker_agent_node(all_mcp_tools)
    safe_tools_node, sensitive_tools_node = get_tool_nodes(all_mcp_tools)
    rag_subgraph_app = get_rag_subgraph()

    # 🌟 ACCEPT CONFIG IN THE PARAMETERS
    async def run_rag_tool(state: WorkerState, config: RunnableConfig):
        last_msg = state["messages"][-1]
        tool_call = last_msg.tool_calls[0]
        tool_call_id = tool_call["id"]
        
        query = tool_call["args"].get("query", "") or tool_call["args"].get("question", "")
        logger.info(f"--- 📚 RUNNING RAG TOOL WRAPPER (Query: '{query}') ---")
        
        # 🌟 THE RAG CLEAN SLATE: Explicitly reset all RagState variables here
        rag_result = await rag_subgraph_app.ainvoke({
            "question": query, 
            "messages": state["messages"],
            "documents": {"vector_facts": [], "graph_facts_used": []}, # Clear old context
            "relevance_score": "",                                     # Clear old scores
            "knowledge_retries": 0,                                    # Reset failure count
            "generation": ""                                           # Clear old answers
        }, config=config) 
        
        final_answer = rag_result.get("generation", "No answer found.")
        logger.info(f"✅ RAG Subgraph finished. Result length: {len(final_answer)} chars.")
        tool_msg = ToolMessage(content=final_answer, tool_call_id=tool_call_id)
        
        return {"messages": [tool_msg]}

    # Build the mini-graph
    builder = StateGraph(WorkerState)
    builder.add_node("worker_agent", worker_agent_node)
    builder.add_node("safe_tools_node", safe_tools_node)
    builder.add_node("sensitive_tools_node", sensitive_tools_node)
    
    # Use the wrapper instead of the raw subgraph
    builder.add_node("rag_subgraph", run_rag_tool)

    builder.add_edge(START, "worker_agent")

    builder.add_edge("safe_tools_node", "worker_agent")
    builder.add_edge("sensitive_tools_node", "worker_agent")
    builder.add_edge("rag_subgraph", "worker_agent")

    def format_worker_output(state: WorkerState):
        """Extracts the final LLM message and pushes it to worker_responses."""
        final_message = state["messages"][-1].content
        task_id = state['task'].get('task_id', 'unknown') # 🌟 FIXED
        logger.info(f"--- 📝 FORMATTING WORKER OUTPUT (Task ID: '{task_id}') ---")
        logger.info(f"📤 Pushed output for Task '{task_id}' to worker_responses.")
        return {"worker_responses": [f"Task {task_id} Result: {final_message}"]}

    builder.add_node("format_output", format_worker_output)
    
    builder.add_conditional_edges(
        "worker_agent",
        route_worker_tools,
        {
            "safe_tools_node": "safe_tools_node",
            "sensitive_tools_node": "sensitive_tools_node",
            "rag_subgraph": "rag_subgraph",
            END: "format_output" 
        }
    )
    
    builder.add_edge("format_output", END)

    return builder.compile(interrupt_before=["sensitive_tools_node"])