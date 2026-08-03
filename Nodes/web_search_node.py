# Nodes/web_search_node.py
from typing import Dict, Any
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

# 🌟 FIX 1: Import the correct Subgraph State
from State.rag_state import RagState 
from Utils.logger import get_logger

logger = get_logger("WEB_SEARCH")

def web_search_node(state: RagState) -> Dict[str, Any]:
    logger.info("--- 🌐 RUNNING WEB SEARCH ---")
    
    # We don't need to check "web_search_approved" because 
    # check_web_search_approval already handled the routing logic!
    
    search_query = state.get("question", "")
    if not search_query.strip():
        logger.warning("Empty search query. Skipping DuckDuckGo search.")
        return {}
        
    logger.info(f"Pinging DuckDuckGo for: '{search_query}'")
    
    try:
        ddg = DuckDuckGoSearchAPIWrapper(max_results=3)
        results = ddg.results(search_query, max_results=3)
        
        web_context = "--- WEB SEARCH RESULTS ---\n"
        found_titles = []
        
        for idx, r in enumerate(results, 1):
            title = r.get("title", "Unknown Site")
            snippet = r.get("snippet", "")
            link = r.get("link", "")
            
            found_titles.append(title)
            # Make the source tag blatantly obvious for the LLM
            web_context += f"[Source: {title} - {link}]\n{snippet}\n\n"

            
        logger.info(f"Retrieved {len(results)} web snippets: {', '.join(found_titles)}")
            
    except Exception as e:
        logger.error(f"DuckDuckGo API Request Failed: {e}")
        web_context = "Web search failed or returned no results."

    # 🌟 FIX 2: Safely append the new string to the existing document string
    existing_docs = state.get("documents", "")
    updated_docs = f"{existing_docs}\n\n{web_context}" 
    
    return {"documents": updated_docs}