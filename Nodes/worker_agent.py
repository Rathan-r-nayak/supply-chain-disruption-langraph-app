from typing import Annotated, Any, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from Schema.task import Task
from State.supply_chain_state import WorkerState
from Utils.logger import get_logger
from Config.llm_config import BASE_URL, API_KEY, client

logger = get_logger("WORKER_AGENT")

WORKER_SYSTEM_PROMPT = """You are the Supply Chain Worker Agent.
Your ONLY job is to execute the "TASK TO EXECUTE" provided by the Orchestrator.

System Context:
The current active user ID is: '{user_id}'. 
If the tool requires a driver_id, use this ID automatically unless explicitly instructed otherwise. If the ID is 'admin', do not pass it as a driver_id.

CRITICAL RULES:
1. Call the ONE necessary tool to perform the action.
2. ONCE A TOOL RETURNS A RESULT (even if it's an error), YOU MUST STOP. DO NOT CALL ANY MORE TOOLS.
3. DO NOT call the exact same tool twice.
4. Once you see the tool response in the history, write a brief text summary of the result so the Orchestrator can read it, and finish.

CITATION RULE:
When writing your final text summary, YOU MUST PRESERVE AND INCLUDE any URLs, links, or 'source_origin' tags provided by the tool. 
- Example: "Trip #102 is active [Source: Internal Trip Database]"
- Example: "Floods reported in Bengaluru ([Deccan Herald](https://link...))"
If you omit the source, the system will fail its audit.
"""

def get_worker_agent_node(all_mcp_tools: list[Any]):
    all_tools = all_mcp_tools 
    
    def worker_agent_node(state: WorkerState, config: RunnableConfig):
        logger.info("--- 👷 RUNNING WORKER AGENT ---")
        
        task = state["task"]
        user_id = config.get("configurable", {}).get("user_id", "Unknown")
        
        logger.info(f"⚡ Executing Task in Parallel: {task.task_id} - {task.description}") 

        fresh_llm = ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-4o-mini", 
            api_key=API_KEY,
            http_client=client,
            temperature=0
        )
        
        llm_with_tools = fresh_llm.bind_tools(all_tools, parallel_tool_calls=False)
        subgraph_messages = state.get("messages", [])

        # Inject the system prompt formatted with the user_id
        final_messages = [
            SystemMessage(content=WORKER_SYSTEM_PROMPT.format(user_id=user_id)),
            HumanMessage(content=f"TASK TO EXECUTE: {state['task'].description}")
        ] + subgraph_messages 
        
        response = llm_with_tools.invoke(final_messages)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            logger.info(f"🛠️ Worker calling tools: {[tc['name'] for tc in response.tool_calls]}")
        else:
            logger.info(f"✅ Worker finished task. Summary: {response.content}")
            
        response.name = "internal_worker"
        return {"messages": [response]}

    return worker_agent_node, all_tools