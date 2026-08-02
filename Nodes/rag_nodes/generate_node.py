from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from Config.llm_config import primary_llm
from State.rag_state import RagState

from Utils.Logger import get_logger

logger = get_logger("GENERATE_NODE")

def generate_node(state: RagState):
    logger.info("--- 🧠 GENERATING FINAL RAG ANSWER ---")
    question = state.get("question", "")
    documents = state.get("documents", {})
    messages = state.get("messages", [])
    retries = state.get("knowledge_retries", 0)

    # 🌟 Format the HybridDocuments dictionary into a clean string for the LLM
    context_str = ""
    if isinstance(documents, dict):
        for f in documents.get("vector_facts", []):
            context_str += f"- {f.get('content', '')}\n"
        for f in documents.get("graph_facts_used", []):
            context_str += f"- {f}\n"
    else:
        # Fallback in case web_search returns a plain string
        context_str = str(documents)

    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a banking policy expert. Answer the user's question using ONLY the provided context. If the context is insufficient, state what is missing."),
        ("human", "Context:\n{documents}\n\nQuestion: {question}")
    ])

    response = (gen_prompt | primary_llm).invoke({
        "documents": context_str, 
        "question": question
    })
    
    final_text = response.content
    
    # Package the response as a ToolMessage
    tool_msg = None
    for msg in reversed(messages):
        if getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                if tc["name"] == "search_bank_policies": 
                    tool_msg = ToolMessage(
                        content=final_text,
                        tool_call_id=tc["id"],
                        name="search_bank_policies"
                    )
                    break
        if tool_msg: break

    return {
        "generation": final_text,
        "messages": [tool_msg] if tool_msg else [],
        "knowledge_retries": retries + 1 # 🌟 Increment to prevent infinite hallucination loops
    }