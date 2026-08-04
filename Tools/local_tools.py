from langchain_core.tools import tool
import os

@tool
def search_logistics_policies(query: str) -> str:
    """Search the internal logistics documentation, company policies, and operational guidelines."""
    # 🌟 We just use 'pass' because LangGraph routes this to the rag_subgraph node instead!
    pass 

@tool
def read_offloaded_file(filepath: str, search_keyword: str = None) -> str:
    """Reads lines from an offloaded file matching an optional search keyword."""
    if not os.path.exists(filepath):
        return "File not found."
    
    matches = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not search_keyword or search_keyword.lower() in line.lower():
                matches.append(line.strip())
                if len(matches) >= 15:  
                    break
                    
    return "\n".join(matches) if matches else "No matching entries found."