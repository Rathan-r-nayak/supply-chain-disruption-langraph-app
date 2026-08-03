# Utils/nba_helper.py
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from Config.llm_config import fast_llm
from Schema.next_best_action_schema import NextBestActionSchema
from Utils.logger import get_logger

logger = get_logger("NBA_HELPER")



def generate_next_best_actions(user_query: str, final_response: str) -> list[str]:
    """Uses fast_llm to predict 2-3 logical next actions for the user."""
    try:
        prompt = PromptTemplate.from_template("""
        You are a logistics assistant predicting the Next Best Action for a supply chain operator or driver.
        
        User's Last Query: {user_query}
        AI Response Given: {final_response}
        
        Suggest exactly 2-3 short, actionable follow-up queries or commands the user might want to run next.
        Keep each suggestion under 8 words. Be extremely relevant to the situation.
        """)
        
        structured_llm = fast_llm.with_structured_output(NextBestActionSchema)
        chain = prompt | structured_llm
        
        result = chain.invoke({
            "user_query": user_query,
            "final_response": final_response[:500]  # Pass truncated context for speed
        })
        
        return result.suggestions
    except Exception as e:
        logger.error(f"Failed to generate Next Best Actions: {e}")
        return ["Check trip details", "View active alerts"]