from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from Config.llm_config import primary_llm

# 🌟 FIX: Revert this import back to RagState
from State.rag_state import RagState 

from Utils.logger import get_logger

logger = get_logger("GENERATE_NODE")

# 🌟 FIX: Change the type hint back to RagState
def generate_node(state: RagState):
    logger.info("--- 🧠 GENERATING FINAL RAG ANSWER ---")
    question = state.get("question", "")
    documents = state.get("documents", {})
    messages = state.get("messages", [])
    retries = state.get("knowledge_retries", 0)

    context_str = ""
    if isinstance(documents, dict):
        for f in documents.get("vector_facts", []):
            context_str += f"- {f.get('content', '')}\n"
        for f in documents.get("graph_facts_used", []):
            context_str += f"- {f}\n"
    else:
        context_str = str(documents)

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a supply chain and logistics expert. Answer the user's question using ONLY the provided context. If the context is insufficient, state what is missing."),
        ("human", "Context:\n{documents}\n\nQuestion: {question}")
    ])

    response = (gen_prompt | primary_llm).invoke({
        "documents": context_str, 
        "question": question
    })
    
    final_text = response.content
    
    tool_msg = None
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                # Update tool name to match your supply chain RAG tool if applicable
                if tc["name"] in ["search_bank_policies", "search_logistics_policies"]: 
                    tool_msg = ToolMessage(
                        content=final_text,
                        tool_call_id=tc["id"],
                        name=tc["name"]
                    )
                    break
        if tool_msg: break

    return {
        "generation": final_text,
        "messages": [tool_msg] if tool_msg else [],
        "knowledge_retries": retries + 1 
    }