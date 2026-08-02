from langchain.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession
from Config.llm_config import fast_llm
from State.supply_chain_state import WorkerState
from Config.llm_config import primary_llm
from Utils.mcp_client import fetch_mcp_tools 
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
import traceback

FASTMCP_SSE_URL = "http://127.0.0.1:8000/mcp/sse"


# Assuming you fetch your tools dynamically from the MCP server here
# tools = mcp_client.get_tools()
banking_tools = [] # Replace with your actual tools list

WORKER_SYSTEM_PROMPT = """You are an autonomous Banking Task Worker.
Your objective is to execute the specific task assigned to you.

Rules:
1. Use the provided tools to fetch required information or execute actions.
2. ALWAYS query the internal knowledge base first.
3. If the knowledge base contains a partial answer, output exactly what you found. Do not say the information is missing just because it is brief.
4. If the internal KB returns NO relevant data, silently use the web search tool to find the answer. Do NOT ask the user for permission to search the web.
5. Output your final answer clearly, using Markdown tables for tabular data if applicable.
"""

# Compile the ReAct subgraph
# react_worker_graph = create_react_agent(
#     model=primary_llm,
#     tools=banking_tools,
#     state_modifier=WORKER_SYSTEM_PROMPT
# )

from Utils.logger import get_logger

logger = get_logger("WORKER_NODE")

def worker_node_function(state: WorkerState):
    tasks = state.get("tasks", [])
    if not tasks:
        raise ValueError("No tasks found in state to process!")
    
    task = tasks[0]
    logger.info(f"⚙️ WORKER STARTED: Executing Task -> {task.task_id}: {task.description}")
    
    try:
        logger.info(f"🔌 Connecting to FastMCP Server at {FASTMCP_SSE_URL}...")

        with sse_client(FASTMCP_SSE_URL) as streams:
            with ClientSession(streams[0], streams[1]) as session:
                session.initialize()
                
                banking_tools = load_mcp_tools(session=session)
                logger.info(f"✅ Successfully loaded {len(banking_tools)} tools.")
                
                react_worker_graph = create_react_agent(
                    model=fast_llm,
                    tools=banking_tools
                )
                
                worker_input = {
                    "messages": [
                        SystemMessage(content=WORKER_SYSTEM_PROMPT),
                        HumanMessage(content=f"Execute this task: {task.description}")
                    ]
                }
                
                result = react_worker_graph.ainvoke(worker_input)
                
        
                final_answer = result["messages"][-1].content
                logger.info(f"✅ WORKER FINISHED: {task.task_id}, final_answer: {final_answer}")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Worker failed on task {task.task_id}:\n{error_details}")
        final_answer = f"Error executing task: {str(e)}"
    
    return {
        "worker_responses": [f"--- Result for {task.task_id} ---\n{final_answer}"]
    }