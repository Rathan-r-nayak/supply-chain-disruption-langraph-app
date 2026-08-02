# from langchain_core.messages import HumanMessage
# from langchain_core.output_parsers import JsonOutputParser
# from langchain_core.prompts import ChatPromptTemplate

# from State.banking_state import BankingState
# from Config.llm_config import fast_llm
# from Utils.Logger import get_logger

# logger = get_logger("KNOWLEDGE_EVAL_NODE")


# def evaluate_node(state: BankingState):
#     logger.info("--- 🔍 EVALUATING KNOWLEDGE RETRIEVAL ---")

#     question = state.get("question", "")
#     messages = state.get("messages", [])

#     if not messages:
#         logger.warning(
#             "⚠️ No messages found in state during knowledge evaluation."
#         )

#         return {
#             "relevance_score": "yes"
#         }

#     last_message = messages[-1]

#     if getattr(last_message, "type", None) != "tool":
#         logger.info(
#             "Last message is not a tool output. "
#             "Defaulting relevance score to 'yes'."
#         )

#         return {
#             "relevance_score": "yes"
#         }

#     retrieved_data = str(last_message.content)

#     logger.info(
#         f"Evaluating retrieved data for question: '{question}'"
#     )

#     grade_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are a strict grader assessing document relevance. "
#                 "Respond in strictly valid JSON: "
#                 "{\"binary_score\": \"yes\"} or "
#                 "{\"binary_score\": \"no\"}. "
#                 "Output 'yes' if the document answers the question, "
#                 "else 'no'.",
#             ),
#             (
#                 "human",
#                 "User question: {question}\n\n"
#                 "Retrieved document:\n{retrieved_data}",
#             ),
#         ]
#     )

#     parser = JsonOutputParser()
#     chain = grade_prompt | fast_llm | parser

#     try:
#         score_dict = chain.invoke(
#             {
#                 "question": question,
#                 "retrieved_data": retrieved_data,
#             }
#         )

#         score = score_dict.get(
#             "binary_score",
#             "yes",
#         ).lower()

#         logger.info(f"Relevance Score: {score.upper()}")

#         return {
#             "relevance_score": score
#         }

#     except Exception as e:
#         logger.error(
#             f"Evaluation failed, defaulting to 'yes': {e}"
#         )

#         return {
#             "relevance_score": "yes"
#         }


# def rewrite_node(state: BankingState):
#     logger.info("--- ✍️ REWRITING KNOWLEDGE QUERY ---")

#     question = state.get("question", "")
#     retries = state.get("knowledge_retries", 0)

#     rewrite_prompt = ChatPromptTemplate.from_messages(
#         [
#             (
#                 "system",
#                 "You are an expert at optimizing search queries "
#                 "for a banking vector database. "
#                 "Look at the original question and output a refined, "
#                 "highly specific search query as plain text.",
#             ),
#             (
#                 "human",
#                 "Original question: {question}",
#             ),
#         ]
#     )

#     rewriter_chain = rewrite_prompt | fast_llm

#     new_query = str(
#         rewriter_chain.invoke(
#             {
#                 "question": question
#             }
#         ).content
#     )

#     logger.info(f"New Query: {new_query}")

#     instruction = (
#         "The previous search yielded irrelevant results. "
#         "Search again using this optimized query: "
#         f"{new_query}"
#     )

#     return {
#         "messages": [
#             HumanMessage(content=instruction)
#         ],
#         "knowledge_retries": retries + 1,
#     }