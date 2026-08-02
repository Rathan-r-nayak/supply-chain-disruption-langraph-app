from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from Config.llm_config import fast_llm

from State.rag_state import RagState
from Utils.Logger import get_logger

logger = get_logger("EVALUATE_NODE")

# 🌟 Updated prompt: It now reads Graph Facts as guiding context!
batch_grade_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict grading assistant. Evaluate the provided document chunks against the user's question.
    
You have been provided with 'Graph Database Facts'. Use these structural facts as absolute domain truth to help you better understand the context and accurately judge which document chunks are actually relevant.

Return a JSON object with a single key 'approved_indices' containing a list of the 0-based indices of the chunks that are relevant.
Example: {{"approved_indices": [0, 2]}}
If none are relevant, return: {{"approved_indices": []}}"""),
    ("human", "Question: {question}\n\nGraph Database Facts (Use as guide):\n{graph_context}\n\nDocument Chunks to Grade:\n{chunks_text}")
])

def evaluate_node(state: RagState):
    logger.info("--- 🔍 EVALUATING KNOWLEDGE RETRIEVAL (GRAPH-GUIDED) ---")
    question = state.get("question", "")
    documents = state.get("documents", {})

    vector_facts = documents.get("vector_facts", []) if isinstance(documents, dict) else []
    graph_facts = documents.get("graph_facts_used", []) if isinstance(documents, dict) else []
    
    if not vector_facts:
        return {"relevance_score": "no", "documents": documents}

    # 1. Format Graph facts into a single string to guide the LLM
    graph_context_str = "No graph facts available."
    if graph_facts:
        graph_context_str = "\n".join([f"- {g}" for g in graph_facts])

    # 2. Format Vector chunks with indices for grading
    chunks_formatted = ""
    for idx, fact in enumerate(vector_facts):
        chunks_formatted += f"\n--- CHUNK {idx} ---\n{fact.get('content', '')}\n"

    try:
        chain = batch_grade_prompt | fast_llm | JsonOutputParser()
        
        # 🌟 Pass BOTH the chunks to grade AND the graph context to guide it
        result = chain.invoke({
            "question": question, 
            "graph_context": graph_context_str,
            "chunks_text": chunks_formatted
        })
        
        approved_indices = result.get("approved_indices", [])
        
        # Keep only the vector chunks the LLM approved
        filtered_vector_facts = [vector_facts[i] for i in approved_indices if i < len(vector_facts)]
        
        logger.info(f"✅ Approved {len(filtered_vector_facts)} out of {len(vector_facts)} chunks using Graph Guidance.")

        filtered_documents = {
            "vector_facts": filtered_vector_facts,
            "graph_facts_used": graph_facts # We keep all graph facts and pass them to the generator
        }

        if len(filtered_vector_facts) > 0 or len(graph_facts) > 0:
            return {"documents": filtered_documents, "relevance_score": "yes"}
        else:
            return {"documents": filtered_documents, "relevance_score": "no"}
            
    except Exception as e:
        logger.error(f"⚠️ Batch evaluation failed, keeping all chunks: {e}")
        return {"documents": documents, "relevance_score": "yes"}