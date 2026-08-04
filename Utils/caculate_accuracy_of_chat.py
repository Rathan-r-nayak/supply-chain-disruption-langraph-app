def calculate_accuracy_score(state_messages: list) -> str:
    """
    Deterministically calculates a confidence score by inspecting tool calls 
    in the message history without extra LLM calls or random numbers.
    """
    if not state_messages:
        return "50% ⚪ (Base Confidence - Direct LLM Generation)"

    tool_names = set()

    for msg in state_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    tool_names.add(tc["name"].lower())
        if hasattr(msg, "name") and msg.name:
            tool_names.add(msg.name.lower())

    mcp_db_tools = {"get_driver_trip_details", "flag_disruptions", "get_locations", "get_trips"}
    rag_tools = {"search_logistics_policies", "rag_subgraph", "generate_node"}
    web_tools = {"fetch_active_hub_news", "web_search_node", "duckduckgo_search"}

    if any(t in tool_names for t in mcp_db_tools):
        return "95% 🟢 (High Confidence - Verified Internal Database)"
    elif any(t in tool_names for t in rag_tools):
        return "85% 🟡 (Medium-High Confidence - Verified Document RAG)"
    elif any(t in tool_names for t in web_tools):
        return "70% 🟠 (Medium Confidence - External Web Search)"
    else:
        return "50% ⚪ (Base Confidence - Direct LLM Generation)"

