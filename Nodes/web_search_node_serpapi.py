import os
from typing import Dict, Any
from langchain_community.utilities import SerpAPIWrapper

# 🌟 FIX 1: Import the correct Subgraph State
from State.rag_state import RagState 
from Utils.logger import get_logger

logger = get_logger("WEB_SEARCH")

def web_search_node(state: RagState) -> Dict[str, Any]:
    logger.info("--- 🌐 RUNNING WEB SEARCH (SERPAPI) ---")
    
    search_query = state.get("question", "")
    if not search_query.strip():
        logger.warning("Empty search query. Skipping SerpApi search.")
        return {}
        
    logger.info(f"Pinging SerpApi for: '{search_query}'")
    
    try:
        # Initialize SerpApi (it will automatically look for SERPAPI_API_KEY in the environment)
        serp_search = SerpAPIWrapper()
        
        # .results() returns the full raw JSON payload from Google Search
        raw_response = serp_search.results(search_query)
        raw_response = {
            "organic_results": [
                {
                    "title": "Test Title", 
                    "snippet": "This is a free test snippet so we don't burn SerpApi credits.", 
                    "link": "https://test.com"
                }
            ]
        }
        
        # We specifically want the standard web links (organic results)
        organic_results = raw_response.get("organic_results", [])
        
        web_context = "--- WEB SEARCH RESULTS ---\n"
        found_titles = []
        
        # Iterate through the top 3 results
        for idx, r in enumerate(organic_results[:3], 1):
            title = r.get("title", "Unknown Site")
            snippet = r.get("snippet", "No snippet available.")
            link = r.get("link", "")
            
            found_titles.append(title)
            # Make the source tag blatantly obvious for the LLM
            web_context += f"[Source: {title} - {link}]\n{snippet}\n\n"

        logger.info(f"Retrieved {len(found_titles)} web snippets: {', '.join(found_titles)}")
            
    except Exception as e:
        logger.error(f"SerpApi Request Failed: {e}")
        web_context = "Web search failed or returned no results."

    # 🌟 FIXED: Safely get the existing HybridDocuments dictionary
    existing_docs = state.get("documents") or {}
    
    # Extract existing lists, defaulting to empty lists if they don't exist yet
    vector_facts = existing_docs.get("vector_facts", [])
    graph_facts = existing_docs.get("graph_facts_used", [])
    
    # Append the new web search data as a properly formatted VectorFact
    if web_context and found_titles:
        vector_facts.append({
            "content": web_context,
            "source": "SerpApi Web Search",
            "score": 1.0
        })
    
    # Return the perfectly preserved dictionary structure
    return {
        "documents": {
            "vector_facts": vector_facts,
            "graph_facts_used": graph_facts
        }
    }