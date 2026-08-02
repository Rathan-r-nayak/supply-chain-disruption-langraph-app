from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from Config.llm_config import fast_llm
from State.rag_state import RagState
from Utils.Logger import get_logger

logger = get_logger("SELF_RAG_REFLECTION")

# ---------------------------------------------------------
# 1. Prompts for Grading
# ---------------------------------------------------------

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict grading assistant. Evaluate whether the generated answer is completely grounded in / supported by the provided documents.
    If the answer contains any facts, numbers, or claims NOT present in the documents, it is an hallucination.
    Return strictly JSON: {{"score": "yes"}} if the answer is completely grounded, or {{"score": "no"}} if it hallucinates."""),
    ("human", "Documents: {documents}\n\nGenerated Answer: {generation}")
])

usefulness_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict grading assistant. Evaluate whether the generated answer is useful and directly answers the user's original question.
    Return strictly JSON: {{"score": "yes"}} if it answers the question, or {{"score": "no"}} if it does not or says it doesn't know."""),
    ("human", "User Question: {question}\n\nGenerated Answer: {generation}")
])

# Initialize the LLM chains
hallucination_grader = hallucination_prompt | fast_llm | JsonOutputParser()
usefulness_grader = usefulness_prompt | fast_llm | JsonOutputParser()

# ---------------------------------------------------------
# 2. The Conditional Router Function
# ---------------------------------------------------------

def check_generation_quality(state: RagState):
    """
    Grades the generation for hallucinations and usefulness.
    Routes to: 'useful', 'not_supported', or 'not_useful'.
    """
    logger.info("--- 🔍 SELF-RAG: CHECKING GENERATION QUALITY ---")
    
    question = state.get("question", "")
    generation = state.get("generation", "")
    documents = state.get("documents", {})
    retries = state.get("knowledge_retries", 0)

    # 🛑 Infinite Loop Safeguard: If we've retried too many times, just accept the answer
    if retries > 3:
        logger.warning("⚠️ Max retries reached. Forcing 'useful' to break loop.")
        return "useful"

    # Format the structured JSON documents into text for the grader LLM
    vector_facts = documents.get("vector_facts", [])
    graph_facts = documents.get("graph_facts_used", [])
    
    docs_text = ""
    for idx, fact in enumerate(vector_facts):
        docs_text += f"[Chunk {idx}]: {fact.get('content', '')}\n"
    for idx, fact in enumerate(graph_facts):
        docs_text += f"[Graph Fact {idx}]: {fact}\n"

    # ---------------------------------------------------------
    # PHASE A: Hallucination / Groundedness Check
    # ---------------------------------------------------------
    try:
        logger.info("🧠 Checking for hallucinations...")
        grounded_result = hallucination_grader.invoke({
            "documents": docs_text, 
            "generation": generation
        })
        
        if grounded_result.get("score", "yes").lower() == "no":
            logger.warning("🚨 Hallucination detected! Routing back to regenerate.")
            return "not_supported"
            
    except Exception as e:
        logger.error(f"⚠️ Hallucination grader failed: {e}. Assuming grounded.")

    # ---------------------------------------------------------
    # PHASE B: Usefulness Check
    # ---------------------------------------------------------
    try:
        logger.info("🧠 Checking if answer is useful to the user...")
        useful_result = usefulness_grader.invoke({
            "question": question, 
            "generation": generation
        })
        
        if useful_result.get("score", "yes").lower() == "no":
            logger.warning("♻️ Answer is grounded but not useful. Routing to rewrite query.")
            return "not_useful"
            
    except Exception as e:
        logger.error(f"⚠️ Usefulness grader failed: {e}. Assuming useful.")

    # ---------------------------------------------------------
    # FINAL: Passed all checks
    # ---------------------------------------------------------
    logger.info("✅ Generation passed all quality checks! Returning to user.")
    return "useful"