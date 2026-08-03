# from typing import Any

# from langchain_core.messages import SystemMessage
# from langgraph.prebuilt import ToolNode, tools_condition

# from State.banking_state import SupplyChainState
# from Config.llm_config import fast_llm
# from Utils.logger import get_logger

# logger = get_logger("KNOWLEDGE_AGENT")

# KNOWLEDGE_SYSTEM_PROMPT = """You are Secure Bank's Knowledge & Policy Specialist.
# Your role is to answer complex queries regarding bank policies, corporate guidelines, loan eligibility, and government incentive schemes (e.g., PMVBRY).

# You have access to a sophisticated GraphRAG knowledge base.

# Rules:
# 1. NEVER guess or hallucinate bank policies. You must ALWAYS use your GraphRAG search tools to retrieve accurate information before answering.
# 2. GraphRAG returns interconnected entities and documents. Synthesize these multiple sources into a clear, unified answer for the user.
# 3. If the retrieved context is insufficient, contradictory, or empty, explicitly state what information is missing and ask the user to clarify. Do not attempt to fill in the blanks yourself.
# 4. Keep your final explanations structured and easy to read.
# """


# def get_knowledge_agent_nodes(all_mcp_tools: list[Any]):
#     """
#     Returns the reasoning node, the tool execution node,
#     and the routing condition.
#     """

#     KNOWLEDGE_TOOL_NAMES = {
#         "query_graphrag",
#         "search_policies",
#         "get_entity_relationships",
#     }

#     knowledge_tools = [
#         t for t in all_mcp_tools
#         if t.name in KNOWLEDGE_TOOL_NAMES
#     ]

#     logger.info(
#         f"📚 Knowledge Agent initialized with {len(knowledge_tools)} tools."
#     )

#     llm_with_tools = fast_llm.bind_tools(knowledge_tools)

#     def knowledge_agent_node(state: SupplyChainState):
#         logger.info("--- 📚 RUNNING KNOWLEDGE AGENT ---")

#         messages = state.get("messages", [])

#         if not messages or not isinstance(messages[0], SystemMessage):
#             messages = [
#                 SystemMessage(content=KNOWLEDGE_SYSTEM_PROMPT)
#             ] + messages

#         # Sanitize messages for Azure OpenAI
#         safe_messages = []

#         for msg in messages:
#             if isinstance(msg.content, list):
#                 msg.content = "\n".join(
#                     str(item)
#                     for item in msg.content
#                 )

#             safe_messages.append(msg)

#         try:
#             response = llm_with_tools.invoke(safe_messages)

#             updates = {
#                 "messages": [response]
#             }

#             if (
#                 not response.tool_calls
#                 and response.content
#             ):
#                 existing_responses = state.get(
#                     "worker_responses",
#                     [],
#                 )

#                 updates["worker_responses"] = (
#                     existing_responses + [response.content]
#                 )

#             return updates

#         except Exception as e:
#             logger.error(
#                 f"❌ Knowledge Agent execution failed: {e}"
#             )
#             raise e

#     knowledge_tools_node = ToolNode(knowledge_tools)

#     return (
#         knowledge_agent_node,
#         knowledge_tools_node,
#         tools_condition,
#     )