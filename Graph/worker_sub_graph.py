from langgraph.graph import StateGraph, START, END
from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig # 🌟 NEW IMPORT

from Graph.rag_sub_graph import get_rag_subgraph
from Nodes.worker_agent import get_worker_agent_node
from Nodes.tool_nodes import get_tool_nodes
from Edges.route_worker_tools import route_worker_tools
from State.supply_chain_state import WorkerState

def get_worker_subgraph(all_mcp_tools):
    """Compiles the worker and its tools into an isolated sub-graph."""
    
    worker_agent_node, _ = get_worker_agent_node(all_mcp_tools)
    safe_tools_node, sensitive_tools_node = get_tool_nodes(all_mcp_tools)
    rag_subgraph_app = get_rag_subgraph()

    # 🌟 ACCEPT CONFIG IN THE PARAMETERS
    async def run_rag_tool(state: WorkerState, config: RunnableConfig):
        last_msg = state["messages"][-1]
        
        # 1. Extract the tool call ID and the query the LLM generated
        tool_call = last_msg.tool_calls[0]
        tool_call_id = tool_call["id"]
        
        # Depending on your tool schema, the argument might be named 'query' or 'question'
        query = tool_call["args"].get("query", "") or tool_call["args"].get("question", "")
        
        # 2. Run the RAG Subgraph with the EXPLICIT question AND THE CONFIG
        # 🌟 PASS CONFIG HERE so the RAG graph shares the same memory/interrupt state
        rag_result = await rag_subgraph_app.ainvoke({
            "question": query, 
            "messages": state["messages"]
        }, config=config) 
        
        # 3. Get the final generated answer
        final_answer = rag_result.get("generation", "No answer found.")
        
        # 4. Format it strictly as a ToolMessage so OpenAI doesn't crash!
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
        return {"worker_responses": [f"Task {state['task'].task_id} Result: {final_message}"]}

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