from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from Tools.local_tools import search_logistics_policies, read_offloaded_file
from State.supply_chain_state import WorkerState
from Utils.logger import get_logger
from Config.llm_config import BASE_URL, API_KEY, client
import os
import uuid
from typing import Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from State.supply_chain_state import WorkerState
from Utils.logger import get_logger
from Config.llm_config import BASE_URL, API_KEY, client
from Utils.message_middleware import offload_large_tool_message, process_tool_message_pipeline


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


logger = get_logger("WORKER_AGENT")


def get_worker_agent_node(all_mcp_tools: list[Any]):
    all_tools = all_mcp_tools 
    
    def worker_agent_node(state: WorkerState, config: RunnableConfig):
        logger.info("--- 👷 RUNNING WORKER AGENT ---")
        
        task = state["task"]
        user_id = config.get("configurable", {}).get("user_id", "Unknown")
        
        logger.info(f"⚡ Executing Task in Parallel: {task.get('task_id')} - {task.get('description')}")

        # fresh_llm = ChatOpenAI(
        #     base_url=BASE_URL,
        #     model="azure/genailab-maas-gpt-4.1", 
        #     api_key=API_KEY,
        #     http_client=client,
        #     temperature=0.0
        # )
        openrouter_key = os.getenv("OPENROUTER_API_KEY")

        fresh_llm = ChatOpenAI(
            openai_api_key=openrouter_key,
            openai_api_base="https://openrouter.ai/api/v1",
            model="openrouter/free", 
            temperature=0.1,
            default_headers={
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Smart Helpdesk Triage App"
            }
        )
        
        llm_with_tools = fresh_llm.bind_tools(all_tools, parallel_tool_calls=False)
        
        # 🌟 STEP 1: Process incoming subgraph messages and offload any large ToolMessages
        raw_subgraph_messages = state.get("messages", [])
        processed_messages = []
        
        for m in raw_subgraph_messages:
            if isinstance(m, ToolMessage):
                processed_messages.append(offload_large_tool_message(m))
                # cleaned_msg = process_tool_message_pipeline(m)
                # processed_messages.append(cleaned_msg)
            else:
                processed_messages.append(m)

        # 🌟 STEP 2: Build final prompt using processed/offloaded messages
        final_messages = [
            SystemMessage(content=WORKER_SYSTEM_PROMPT.format(user_id=user_id)),
            HumanMessage(content=f"TASK TO EXECUTE: {task.get('description')}")
        ] + processed_messages 
        
        response = llm_with_tools.invoke(final_messages)
        
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_names = [tc['name'] for tc in response.tool_calls]
            logger.info(f"🛠️ Worker calling tools: {tool_names}")
            
            return {
            "messages": [response], 
                "executed_tools": tool_names
            }
        else:
            logger.info(f"✅ Worker finished task. Summary: {response.content}")
            response.name = "internal_worker" 
            
            return {
                "messages": [response],
                "executed_tools": []
            }

    return worker_agent_node, all_tools