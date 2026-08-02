from typing import Any
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from State.banking_state import BankingState
from Utils.Logger import get_logger
from Config.llm_config import BASE_URL, API_KEY

logger = get_logger("MASTER_AGENT")

# 🌟 1. The RAG Dummy Tool
@tool
def search_bank_policies(query: str) -> str:
    """
    Search Secure Bank's policy documents, loan terms, interest rates, rules, and FAQs.
    ALWAYS use this tool if the user asks a general banking knowledge question.
    """
    # We use 'pass' because this function is never actually executed.
    # The router edge intercepts this tool call and diverts it to the Self-RAG loop!
    pass

MASTER_SYSTEM_PROMPT = """You are Secure Bank's Master AI Assistant.
Your job is to assist customers by dynamically selecting the right tools.

Rules:
1. For checking balances or fetching accounts, use standard account tools.
2. For transferring money or paying bills, use sensitive transaction tools.
3. For general knowledge, policies, loans, or interest rates, ALWAYS use the 'search_bank_policies' tool.
4. If a user asks a combined question (e.g., "What is my balance and what are your loan rates?"), call BOTH tools simultaneously.
5. Always maintain a polite, secure tone.
"""

def get_master_agent_nodes(all_mcp_tools: list[Any]):
    # Combine live MCP tools with our Self-RAG dummy tool
    all_tools = all_mcp_tools + [search_bank_policies]
    
    logger.info(f"🧠 Master Agent initialized with {len(all_tools)} total tools.")

    async def master_agent_node(state: BankingState):
        logger.info("--- 🧠 RUNNING MASTER AGENT ---")

        # 🌟 Firewall-Safe Async LLM Initialization
        fresh_llm = ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-4o", # Use 4o for superior tool calling
            api_key=API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0),
            http_async_client=httpx.AsyncClient(verify=False, timeout=120.0),
            temperature=0
        )

        llm_with_tools = fresh_llm.bind_tools(all_tools)

        messages = state.get("messages", [])
        
        # Inject system prompt at the start of the conversation
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=MASTER_SYSTEM_PROMPT)] + messages

        try:
            # The LLM evaluates the messages and either replies or generates tool calls
            response = await llm_with_tools.ainvoke(messages)
            
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(f"🛠️ Master Agent requested tools: {tool_names}")
            
            # Wipe old worker responses and append the new AI message
            return {
                "messages": [response],
                "worker_responses": []
            }

        except Exception as e:
            logger.error(f"❌ Master Agent execution failed: {e}")
            raise e

    return master_agent_node, all_tools