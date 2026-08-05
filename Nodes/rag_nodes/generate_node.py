from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage
from Config.llm_config import synthesis_llm
from State.rag_state import RagState 
from Utils.logger import get_logger

logger = get_logger("GENERATE_NODE")

def generate_node(state: RagState):
    logger.info("--- 🧠 GENERATING FINAL RAG ANSWER WITH CITATIONS ---")
    question = state.get("question", "")
    documents = state.get("documents", {})
    messages = state.get("messages", [])
    retries = state.get("knowledge_retries", 0)

    context_str = ""
    if isinstance(documents, dict):
        # 🌟 1. Inject the explicit source for Vector Facts
        for f in documents.get("vector_facts", []):
            content = f.get('content', '')
            source = f.get('source', 'Unknown Source')
            context_str += f"- [Source: {source}] {content}\n"
            
        # 🌟 2. Assign a default source for Graph Facts
        for f in documents.get("graph_facts_used", []):
            context_str += f"- [Source: Internal Knowledge Graph] {f}\n"
    else:
        context_str = str(documents)

    # 🌟 3. Update the System Prompt with strict citation rules
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a supply chain and logistics expert. Answer the user's question using ONLY the provided context.\n\n"
            "CITATION RULE:\n"
            "You MUST cite your sources directly in the text. Whenever you state a fact derived from the context, "
            "append the exact source tag provided in brackets at the end of the sentence.\n"
            "Example: 'The standard transit time to Hub B is 4 days [Source: transit_policy_2024.pdf].'\n\n"
            "If the context is insufficient to fully answer the question, clearly state what information is missing."
        )),
        ("human", "Context:\n{documents}\n\nQuestion: {question}")
    ])

    response = (gen_prompt | synthesis_llm).invoke({
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