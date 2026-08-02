# import asyncio
# import json
# from contextlib import asynccontextmanager
# from fastapi import FastAPI, Request
# from fastapi.responses import StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import os
# import tempfile
# from fastapi import FastAPI, Request, File, Form, UploadFile
# from fastapi.responses import StreamingResponse
# from mcp.client.sse import sse_client
# from mcp.client.session import ClientSession
# from langchain_mcp_adapters.tools import load_mcp_tools
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
# from langchain_core.messages import HumanMessage

# # Import project modules
# from main import build_graph
# from Utils.Logger import get_logger

# logger = get_logger("SERVER_API")


# # phoenix implementation
# import phoenix as px
# from openinference.instrumentation.langchain import LangChainInstrumentor
# from opentelemetry import trace as trace_api
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# from opentelemetry.sdk import trace as trace_sdk
# from opentelemetry.sdk.trace.export import SimpleSpanProcessor

# try:
#     from Utils.database_manager import ltm_store
# except ImportError:
#     ltm_store = None
    
# logger.info("🔥 Starting Arize Phoenix for telemetry...")
# # px.launch_app()  # Launches local Phoenix dashboard on http://localhost:6006

# # Configure OpenTelemetry to export spans to the Phoenix server
# endpoint = "http://127.0.0.1:6006/v1/traces"
# tracer_provider = trace_sdk.TracerProvider()
# tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
# trace_api.set_tracer_provider(tracer_provider)

# # Instrument LangChain/LangGraph under the hood
# LangChainInstrumentor().instrument()
# logger.info("✅ LangChain Instrumentor active. Traces will appear in Phoenix.")



# # Global state for MCP
# mcp_session = None
# mcp_tools = []
# MCP_SERVER_URL = "http://localhost:8000/mcp/sse"

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Establishes a persistent connection to the FastMCP server on boot."""
#     global mcp_session, mcp_tools
#     logger.info("🔌 Connecting to FastMCP Server...")
#     try:
#         async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
#             async with ClientSession(read_stream, write_stream) as session:
#                 await session.initialize()
#                 mcp_tools = await load_mcp_tools(session)
#                 mcp_session = session
#                 logger.info(f"✅ FastMCP Connected! Loaded {len(mcp_tools)} tools.")
#                 yield
#     except Exception as e:
#         logger.warning(f"⚠️ FastMCP Connection skipped/failed: {e}")
#         yield
#     logger.info("🛑 Disconnected from FastMCP.")

# app = FastAPI(title="Banking Agent API", lifespan=lifespan)

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class ChatRequest(BaseModel):
#     thread_id: str
#     user_id: str
#     message: str | None = None
#     action: str | None = None  # "approve" or "reject" for HITL

# import traceback # 🌟 Add this at the top of your server.py file if not there

# # @app.post("/chat/stream")
# # async def chat_stream(req: ChatRequest):
# #     """Executes the graph and streams standard SSE tokens using the stable messages mode."""
    
# #     logger.info(f"💬 [INCOMING REQUEST] Thread: '{req.thread_id}' | User: '{req.user_id}' | Message: '{req.message}' | Action: '{req.action}'")

# #     async def generate():
# #         full_streamed_response = []
        
# #         async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
# #             graph = build_graph(all_mcp_tools=mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
            
# #             thread_config = {"configurable": {"thread_id": req.thread_id, "user_id": req.user_id}}
            
# #             # Handle Human-in-the-Loop Resumes
# #             if req.action == "approve":
# #                 inputs = None
# #             elif req.action == "reject":
# #                 cancellation_msg = "Transaction cancelled by user."
# #                 logger.info(f"🛑 [STREAM END] Thread: '{req.thread_id}' | Output: {cancellation_msg}")
# #                 yield f"data: {cancellation_msg}\n\n"
# #                 return
# #             else:
# #                 inputs = {
# #                     "question": req.message,
# #                     "worker_responses": [],
# #                     "messages": [HumanMessage(content=req.message)]
# #                 }

# #             try:
# #                 # 1. Stream the messages normally
# #                 async for msg, metadata in graph.astream(inputs, config=thread_config, stream_mode="messages"):
                    
# #                     node_name = metadata.get("langgraph_node")
                    
# #                     if msg.content and isinstance(msg.content, str):
# #                         content_clean = msg.content.replace('\n', '\\n')
                        
# #                         if node_name == "worker_agent":
# #                             yield f"event: thought\ndata: {content_clean}\n\n"
                            
# #                         elif node_name == "aggregator":
# #                             yield f"event: message\ndata: {content_clean}\n\n"

# #                 # 🌟 2. NEW: Check if the graph paused at a security gate!
# #                 # If it did, send the interrupt signal so the UI instantly shows the buttons.
# #                 state = await graph.aget_state(thread_config)
# #                 if len(state.next) > 0:
# #                     yield f"event: message\ndata: __INTERRUPT__\n\n"

# #             except Exception as e:
# #                 logger.error("❌ Stream Exception caught!")
# #                 traceback.print_exc() 
# #                 yield f"event: error\ndata: An internal error occurred.\n\n"

# #     return StreamingResponse(generate(), media_type="text/event-stream")



# @app.post("/chat/stream")
# async def chat_stream(
#     thread_id: str = Form(...),
#     user_id: str = Form(...),
#     message: str = Form(None),
#     action: str = Form(None),
#     image: UploadFile = File(None)  # 🌟 NEW: Accepts the image file
# ):
#     """Executes the graph and streams standard SSE tokens using the stable messages mode."""
    
#     logger.info(f"💬 [INCOMING REQUEST] Thread: '{thread_id}' | User: '{user_id}' | Message: '{message}' | Action: '{action}'")

#     async def generate():
#         image_path = None
        
#         # 🌟 1. Save the uploaded image to a temporary file
#         if image and image.filename:
#             logger.info(f"📸 Image uploaded: {image.filename}")
#             # Create a temporary file path
#             fd, temp_path = tempfile.mkstemp(suffix=f"_{image.filename}")
#             with os.fdopen(fd, 'wb') as f:
#                 content = await image.read()
#                 f.write(content)
#             image_path = temp_path  # We will pass this to the graph

#         try:
#             async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
#                 graph = build_graph(all_mcp_tools=mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
                
#                 thread_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
                
#                 # Handle Human-in-the-Loop Resumes
#                 if action == "approve":
#                     inputs = None
#                 elif action == "reject":
#                     cancellation_msg = "Transaction cancelled by user."
#                     logger.info(f"🛑 [STREAM END] Thread: '{thread_id}' | Output: {cancellation_msg}")
#                     yield f"data: {cancellation_msg}\n\n"
#                     return
#                 else:
#                     inputs = {
#                         "question": message or "",
#                         "worker_responses": [],
#                         "messages": [HumanMessage(content=message)] if message else [],
#                         "image_path": image_path  # 🌟 2. Pass the image path into your SupplyChainState
#                     }

#                 # Stream the messages normally
#                 async for msg, metadata in graph.astream(inputs, config=thread_config, stream_mode="messages"):
#                     node_name = metadata.get("langgraph_node")
                    
#                     if msg.content and isinstance(msg.content, str):
#                         content_clean = msg.content.replace('\n', '\\n')
                        
#                         if node_name == "worker_agent":
#                             yield f"event: thought\ndata: {content_clean}\n\n"
                            
#                         # Ensure we stream the output_guardrail (which is your final output now)
#                         elif node_name in ["aggregator", "output_guardrail", "triage_router"]:
#                             yield f"event: message\ndata: {content_clean}\n\n"

#                 # Check if the graph paused at a security gate!
#                 state = await graph.aget_state(thread_config)
#                 if len(state.next) > 0:
#                     yield f"event: message\ndata: __INTERRUPT__\n\n"

#         except Exception as e:
#             logger.error("❌ Stream Exception caught!")
#             traceback.print_exc() 
#             yield f"event: error\ndata: An internal error occurred.\n\n"
            
#         finally:
#             # 🌟 3. CLEANUP: Delete the temp image file so your server doesn't run out of storage!
#             if image_path and os.path.exists(image_path):
#                 os.remove(image_path)
#                 logger.info(f"🗑️ Cleaned up temporary image file: {image_path}")

#     return StreamingResponse(generate(), media_type="text/event-stream")

# @app.get("/chat/history")
# async def get_history(thread_id: str):
#     async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
#         graph = build_graph(all_mcp_tools=[], checkpointer=checkpointer, ltm_store=ltm_store)
#         state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
        
#         if not state or not state.values:
#             return {"messages": [], "is_paused": False}
            
#         # 🌟 FIX: We must just check if it's paused AT ALL!
#         # The parent graph sees "worker_subgraph" in state.next, not the internal tool node.
#         is_paused = len(state.next) > 0 
        
#         formatted_msgs = []
#         for msg in state.values.get("messages", []):
#             if getattr(msg, 'type', '') in ['human', 'ai'] and getattr(msg, 'content', ''):
                
#                 if msg.type == 'ai' and getattr(msg, 'tool_calls', None):
#                     continue
                    
#                 # 🌟 SEPARATE THOUGHTS FROM FINAL ANSWERS
#                 if msg.type == 'ai' and getattr(msg, 'name', '') == 'internal_worker':
#                     formatted_msgs.append({"role": "thought", "content": msg.content})
#                 else:
#                     formatted_msgs.append({"role": "user" if msg.type == 'human' else "assistant", "content": msg.content})
                
#         return {"messages": formatted_msgs, "is_paused": is_paused}