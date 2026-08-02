# import asyncio
# import sqlite3
# from mcp.client.sse import sse_client
# from mcp.client.session import ClientSession
# from langchain_mcp_adapters.tools import load_mcp_tools 
# from langgraph.checkpoint.sqlite import SqliteSaver


# # Import your custom SQLite LTM Store manager class
# # (Assuming you saved the SqliteKeyValueStore class in a file named database_manager.py)
# # If it's in the same file or another module, adjust the import accordingly:
# try:
#     from database_manager import ltm_store
# except ImportError:
#     # Fallback definition if inline
#     import json
#     class SqliteKeyValueStore:
#         def __init__(self, db_path="banking_ltm.db"):
#             self.conn = sqlite3.connect(db_path, check_same_thread=False)
#             self._create_table()
#         def _create_table(self):
#             with self.conn:
#                 self.conn.execute("""
#                     CREATE TABLE IF NOT EXISTS memories (
#                         namespace TEXT, key TEXT, value TEXT, PRIMARY KEY (namespace, key)
#                     )
#                 """)
#         def put(self, namespace: tuple, key: str, value: dict):
#             ns_str = ".".join(namespace)
#             with self.conn:
#                 self.conn.execute("INSERT OR REPLACE INTO memories (namespace, key, value) VALUES (?, ?, ?)", 
#                                   (ns_str, key, json.dumps(value)))
#         def search(self, namespace: tuple):
#             ns_str = ".".join(namespace)
#             cursor = self.conn.cursor()
#             cursor.execute("SELECT key, value FROM memories WHERE namespace = ?", (ns_str,))
#             class Item:
#                 def __init__(self, val): self.value = val
#             return [Item(json.loads(row[1])) for row in cursor.fetchall()]
    
#     ltm_store = SqliteKeyValueStore()

# # Import your updated graph builder function from main.py
# from main import build_graph 

# async def run_banking_agent():
#     """Starts the LangGraph app, connects to FastMCP, and attaches SQLite persistence + LTM store."""
    
#     MCP_SERVER_URL = "http://localhost:8000/mcp/sse" 
    
#     print("🔌 Connecting to FastMCP Server...")
    
#     async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
#         async with ClientSession(read_stream, write_stream) as session:
            
#             await session.initialize()
#             print("✅ Connected! Fetching tools...")
            
#             all_mcp_tools = await load_mcp_tools(session)
#             print(f"Loaded {len(all_mcp_tools)} tools.")
            
#             # 1. Initialize SQLite Checkpointer for short-term thread persistence
#             sqlite_conn = sqlite3.connect("banking_checkpoints.db", check_same_thread=False)
#             checkpointer = SqliteSaver(sqlite_conn)
            
#             # 2. Build graph with tools, checkpointer, and permanent LTM store
#             app = build_graph(all_mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
            
#             print("🚀 Banking Agent Ready! Type 'exit' to quit.")
#             print("-" * 50)
            
#             # Config includes both thread tracking and user_id for LTM namespace mapping
#             thread_config = {
#                 "configurable": {
#                     "thread_id": "user_thread_123",
#                     "user_id": "rathan_123"  # <--- Used by recall_node and remember_node
#                 }
#             }
            
#             while True:
#                 user_input = input("\nYou: ").strip()
#                 if user_input.lower() in ['exit', 'quit']:
#                     break
                
#                 # Check if graph is currently paused waiting for HITL approval
#                 state = app.get_state(thread_config)
                
#                 if state.next and "sensitive_tools_node" in state.next:
#                     if user_input.lower() in ["yes", "y", "approve", "confirm"]:
#                         print("⚡ Resuming graph execution (User Approved)...")
#                         inputs = None  # Passing None unpauses the graph
#                     else:
#                         print("❌ Transaction cancelled by user.")
#                         continue
#                 else:
#                     # Normal message payload mapping to state inputs
#                     inputs = {
#                         "messages": [("user", user_input)],
#                         "question": user_input,
#                         "worker_responses": []
#                     }

#                 # Stream execution updates through the graph nodes
#                 async for event in app.astream(inputs, thread_config, stream_mode="updates"):
#                     for node_name, node_output in event.items():
#                         if node_name == "aggregator" and "generation" in node_output:
#                             print(f"\nAgent: {node_output['generation']}")
#                         elif node_name == "__interrupt__":
#                             print("\n TRANSACTION APPROVAL REQUIRED!")
#                             print("Type 'yes' to approve the transfer, or anything else to cancel.")
#                         elif node_name == "remember_node":
#                             pass # Background task completed silently
                            
#     print("\n🛑 Disconnected from FastMCP Server.")

# if __name__ == "__main__":
#     asyncio.run(run_banking_agent())