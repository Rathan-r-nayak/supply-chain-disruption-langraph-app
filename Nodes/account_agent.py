from typing import Any
import httpx
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode, tools_condition

from State.supply_chain_state import SupplyChainState
from Utils.logger import get_logger
from Config.llm_config import BASE_URL, API_KEY

logger = get_logger("ACCOUNT_AGENT")

ACCOUNT_SYSTEM_PROMPT = """You are Secure Bank's Account Specialist.
Your job is to assist customers with account management tasks ONLY.

Available capabilities:
- Checking account balances
- Opening new savings or checking accounts
- Fetching account details
- Processing deposits and withdrawals

Rules:
1. Always maintain a polite, clear, and secure tone.
2. If an operation succeeds or fails, explain the outcome concisely.
3. If the user asks to transfer money to another person, you MUST politely inform them that you cannot do that and ask them to repeat the request so the Orchestrator can route them to the Transaction Agent."""

def get_account_agent_nodes(all_mcp_tools: list[Any]):
    ACCOUNT_TOOL_NAMES = {
        "create_new_account",
        "check_balance",
        "get_account",
        "get_all_accounts",
        "deposit_money",
        "withdraw_money"
    }

    account_tools = [t for t in all_mcp_tools if t.name in ACCOUNT_TOOL_NAMES]
    logger.info(f"💳 Account Agent initialized with {len(account_tools)} tools.")

    async def account_agent_node(state: SupplyChainState):
        logger.info("--- 💳 RUNNING ACCOUNT AGENT ---")

        # Instantiate fresh LLM & HTTP client inside the active async event loop
        fresh_llm = ChatOpenAI(
            base_url=BASE_URL,
            model="azure/genailab-maas-gpt-4o-mini",
            api_key=API_KEY,
            http_client=httpx.Client(verify=False, timeout=120.0),
            http_async_client=httpx.AsyncClient(
                verify=False,
                timeout=120.0
            ),
            temperature=0
        )

        llm_with_tools = fresh_llm.bind_tools(account_tools)

        messages = state.get("messages", [])

        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=ACCOUNT_SYSTEM_PROMPT)] + messages

        try:
            response = await llm_with_tools.ainvoke(messages)
            updates = {"messages": [response]}

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_names = [tc["name"] for tc in response.tool_calls]
                logger.info(
                    f"🛠️ Account Agent requested tool call(s): {tool_names}"
                )

            elif response.content:
                snippet = str(response.content)[:100].replace("\n", " ")
                logger.info(
                    f"💬 Account Agent generated final response: '{snippet}...'"
                )

                existing_responses = state.get("worker_responses", [])
                updates["worker_responses"] = (
                    existing_responses + [str(response.content)]
                )
                logger.info(
                    "✅ Pushed Account Agent final response to worker_responses."
                )

            return updates

        except Exception as e:
            logger.error(f"❌ Account Agent execution failed: {e}")
            raise e

    account_tools_node = ToolNode(account_tools)
    return account_agent_node, account_tools_node, tools_condition