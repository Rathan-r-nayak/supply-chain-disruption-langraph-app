from typing import Annotated, Any, TypedDict
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool

from Schema.task import Task
from State.banking_state import WorkerState
from Utils.Logger import get_logger
from Config.llm_config import BASE_URL, API_KEY, client
from Utils.helpers import get_short_term_memory
from langgraph.graph.message import add_messages

logger = get_logger("WORKER_AGENT")

# 🌟 Define the specific state this parallel node receives

@tool
def search_bank_policies(query: str) -> str:
    """Search Secure Bank's policy documents, loan terms, interest rates, rules, and FAQs."""
    pass

WORKER_SYSTEM_PROMPT = """You are the Banking Worker Agent.
Your ONLY job is to execute the "TASK TO EXECUTE" provided by the Orchestrator.

CRITICAL RULES:
1. Call the ONE necessary tool to perform the action.
2. ONCE A TOOL RETURNS A RESULT (even if it's an error), YOU MUST STOP. DO NOT CALL ANY MORE TOOLS.
3. DO NOT try to "verify" a transaction by checking balances afterwards.
4. DO NOT call the exact same tool twice.
5. Once you see the tool response in the history, write a brief text summary of the result so the Orchestrator can read it, and finish.
"""

# In Nodes/worker_agent.py

def get_worker_agent_node(all_mcp_tools: list[Any]):
    all_tools = all_mcp_tools + [search_bank_policies]
    
    def worker_agent_node(state: WorkerState):
        logger.info("--- 👷 RUNNING WORKER AGENT ---")
        
        task = state["task"]
        logger.info(f"⚡ Executing Task in Parallel: {task.task_id} - {task.description}") 

        fresh_llm = ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-4o-mini", 
            api_key=API_KEY,
            http_client=client,
            temperature=0
        )
        
        # 🌟 FIX: Force the LLM to only pick one tool at a time!
        llm_with_tools = fresh_llm.bind_tools(all_tools, parallel_tool_calls=False)
        
        # 🌟 FIX: We MUST include the messages from the current subgraph state!
        # If a tool just ran, its output will be in state["messages"]. 
        # If we ignore it, the LLM will just call the tool again.
        subgraph_messages = state.get("messages", [])

        final_messages = [
            SystemMessage(content=WORKER_SYSTEM_PROMPT),
            HumanMessage(content=f"TASK TO EXECUTE: {state['task'].description}")
        ] + subgraph_messages # Append the subgraph's ongoing conversation
        
        response = llm_with_tools.invoke(final_messages)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"🛠️ Worker calling tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            # If no tools were called, it must be summarizing the result
            logger.info(f"✅ Worker finished task. Summary: {response.content}")
            
        response.name = "internal_worker"
        return {"messages": [response]}

    return worker_agent_node, all_tools