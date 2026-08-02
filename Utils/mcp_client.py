import os
import httpx
import streamlit as st
from Utils.logger import get_logger
import asyncio
from mcp.client.sse import sse_client
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession



FASTMCP_SSE_URL = os.getenv("FASTMCP_SSE_URL", "http://localhost:8000/mcp/sse")


logger = get_logger("APIRESPONSE")
_CACHED_TOOLS = None


async def fetch_mcp_tools():
    """Fetches the configuration lookup catalog directly from the FastAPI backend."""
    
    global _CACHED_TOOLS
    
    if _CACHED_TOOLS is not None:
        return _CACHED_TOOLS
    
    logger.info(f"🔌 Connecting to FastMCP Server at {FASTMCP_SSE_URL}...")
    
    try:
        async with sse_client(FASTMCP_SSE_URL) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                
                logger.info(f"✅ Successfully loaded {len(tools)} tools from FastMCP.")
                _CACHED_TOOLS = tools
                return _CACHED_TOOLS
                
    except Exception as e:
        logger.error(f"⚠️ Failed to connect to backend catalog service: {e}")