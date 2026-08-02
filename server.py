import asyncio
import json
import os
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TypedDict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel
from rag.ingestion.pipeline import ingest_file

# Import project modules
from main import build_graph
from Utils.Logger import get_logger

logger = get_logger("SERVER_API")

# phoenix implementation
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

try:
    from Utils.database_manager import ltm_store
except ImportError:
    ltm_store = None

logger.info("🔥 Starting Arize Phoenix for telemetry...")
# px.launch_app()  # Launches local Phoenix dashboard on http://localhost:6006

# Configure OpenTelemetry to export spans to the Phoenix server
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider)

# Instrument LangChain/LangGraph under the hood
LangChainInstrumentor().instrument()
logger.info("✅ LangChain Instrumentor active. Traces will appear in Phoenix.")

# =============================================================================
# GLOBAL STATE & FASTAPI STARTUP / LIFESPAN
# =============================================================================
mcp_session = None
mcp_tools = []
MCP_SERVER_URL = "http://localhost:8000/mcp/sse"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Establishes a persistent connection to the FastMCP server on boot."""
    global mcp_session, mcp_tools
    logger.info("🔌 Connecting to FastMCP Server...")
    try:
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                mcp_session = session
                logger.info(f"✅ FastMCP Connected! Loaded {len(mcp_tools)} tools.")
                yield
    except Exception as e:
        logger.warning(f"⚠️ FastMCP Connection skipped/failed: {e}")
        yield
    logger.info("🛑 Disconnected from FastMCP.")

# 🌟 CRITICAL FIX: Pass the lifespan function into FastAPI here
app = FastAPI(title="Banking Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# DATABASE & USERS SETUP
# =============================================================================
USERS_DB: dict[str, dict] = {
    "admin1": {"password": "admin123", "role": "ADMIN"},
    "cust1":  {"password": "cust123",  "role": "CUSTOMER"},
    "rathan":  {"password": "1234",  "role": "CUSTOMER"},
}

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

METADATA_DIR = "metadata"
os.makedirs(METADATA_DIR, exist_ok=True)
THREADS_METADATA_FILE = os.path.join(METADATA_DIR, "threads.json")
DOCS_METADATA_FILE = os.path.join(METADATA_DIR, "docs.json")

def _read_docs_metadata() -> list[dict]:
    if not os.path.exists(DOCS_METADATA_FILE):
        _write_docs_metadata([])   # first run - create an empty metadata file
        return []
    with open(DOCS_METADATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def _write_docs_metadata(docs: list[dict]) -> None:
    with open(DOCS_METADATA_FILE, "w") as f:
        json.dump(docs, f, indent=2)

def _read_threads_metadata() -> dict[str, dict]:
    if not os.path.exists(THREADS_METADATA_FILE):
        _write_threads_metadata({})   # first run - create an empty metadata file
        return {}
    with open(THREADS_METADATA_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}

def _write_threads_metadata(threads: dict[str, dict]) -> None:
    with open(THREADS_METADATA_FILE, "w") as f:
        json.dump(threads, f, indent=2)

# =============================================================================
# MODELS
# =============================================================================
class LoginRequest(BaseModel):
    user_id: str
    password: str

class LoginResponse(BaseModel):
    user_id: str
    role: str

class ThreadSummary(TypedDict):
    thread_id: str
    title: str
    created_at: str

class DocumentInfo(TypedDict):
    filename: str
    uploaded_by: str
    uploaded_at: str

class UploadResponse(BaseModel):
    filename: str
    status: str
    chunks_ingested: int | None = None



# --- Add these imports at the top of server.py ---
from langchain_core.prompts import PromptTemplate
from Config.llm_config import fast_llm  # Ensure this points to your fast LLM

# --- Paste this function anywhere above your endpoints ---
import threading

def update_thread_title_background(thread_id: str, user_msg: str, ai_msg: str):
    """Runs in a completely separate thread to avoid FastAPI connection closures."""
    try:
        # 🌟 Quick check: If the user message is just a basic greeting, skip the LLM and set a clean name!
        greetings = ["hi", "hello", "hey", "hii", "greetings", "good morning", "good evening"]
        cleaned_msg = user_msg.strip().lower()
        
        if cleaned_msg in greetings:
            new_title = "General Inquiry"
        else:
            prompt = PromptTemplate.from_template(
                "You are a helpful assistant that creates very short, concise titles for chat conversations.\n"
                "Based on the user's request and the AI's response, create a title that is AT MOST 20 characters long.\n"
                "Do not use quotes. Just output the title.\n\n"
                "User: {user_msg}\n"
                "AI: {ai_msg}"
            )
            
            chain = prompt | fast_llm
            response = chain.invoke({"user_msg": user_msg, "ai_msg": ai_msg[:500]}) 
            
            new_title = response.content.strip().strip('"').strip("'")
            if len(new_title) > 20:
                new_title = new_title[:17] + "..."
            
        threads = _read_threads_metadata()
        if thread_id in threads:
            threads[thread_id]["title"] = new_title
            threads[thread_id]["is_renamed"] = True  
            _write_threads_metadata(threads)
            logger.info(f"📝 Thread {thread_id} automatically retitled to: '{new_title}'")
            
    except Exception as e:
        logger.error(f"❌ Failed to generate background title: {e}")
# =============================================================================
# ENDPOINTS
# =============================================================================
@app.post("/auth/login", response_model=LoginResponse)
async def login(req: LoginRequest):
    user = USERS_DB.get(req.user_id)
    if not user or user["password"] != req.password:
        raise HTTPException(status_code=401, detail="Invalid user_id or password")
    return LoginResponse(user_id=req.user_id, role=user["role"])

@app.post("/chat/stream")
async def chat_stream(
    thread_id: str = Form(...),
    user_id: str = Form(...),
    message: str = Form(None),
    action: str = Form(None),
    image: UploadFile = File(None)
):
    """Executes the graph and streams standard SSE tokens using the stable messages mode."""

    logger.info(f"💬 [INCOMING REQUEST] Thread: '{thread_id}' | User: '{user_id}' | Message: '{message}' | Action: '{action}'")

    async def generate():
        image_path = None
        final_ai_text = ""     # Store the AI response for the titler
        
        if image and image.filename:
            logger.info(f"📸 Image uploaded: {image.filename}")
            fd, temp_path = tempfile.mkstemp(suffix=f"_{image.filename}")
            with os.fdopen(fd, 'wb') as f:
                content = await image.read()
                f.write(content)
            image_path = temp_path

        try:
            async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
                graph = build_graph(all_mcp_tools=mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
                thread_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

                if action == "approve":
                    inputs = None
                elif action == "reject":
                    cancellation_msg = "Transaction cancelled by user."
                    yield f"data: {cancellation_msg}\n\n"
                    return
                else:
                    inputs = {
                        "question": message or "",
                        "worker_responses": [],
                        "messages": [HumanMessage(content=message)] if message else [],
                        "image_path": image_path 
                    }
                    if message:
                        register_thread(thread_id, user_id, message)

                # STREAMING LOOP
                # STREAMING LOOP
                async for mode, payload in graph.astream(inputs, config=thread_config, stream_mode=["messages", "updates"]):
                    
                    if mode == "messages":
                        msg, metadata = payload
                        node_name = metadata.get("langgraph_node", "unknown")

                        if msg.content and isinstance(msg.content, str):
                            content_clean = msg.content.replace('\n', '\\n')

                            # We STILL stream the worker_agent's thoughts so the user sees progress
                            if node_name == "worker_agent":
                                yield f"event: thought\ndata: {content_clean}\n\n"
                            
                            # ❌ We intentionally DO NOT stream the "aggregator" here anymore.
                            # We must wait for the output_guardrail to scrub the data first!

                    elif mode == "updates":
                        for node_name, state_update in payload.items():
                            
                            # 🌟 ADDED 'output_guardrail' TO THIS LIST
                            if node_name in ["input_guardrail", "triage_router", "output_guardrail"]:
                                out_msg = None
                                
                                if state_update.get("messages"):
                                    last_msg = state_update["messages"][-1]
                                    if getattr(last_msg, "type", "") == "ai":
                                        out_msg = last_msg.content
                                elif state_update.get("generation"):
                                    out_msg = state_update["generation"]

                                if out_msg:
                                    final_ai_text = out_msg  
                                    content_clean = out_msg.replace('\n', '\\n')
                                    # This yields the ENTIRE, fully redacted message all at once!
                                    yield f"event: message\ndata: {content_clean}\n\n"

                # Check if paused at security gate
                state = await graph.aget_state(thread_config)
                if len(state.next) > 0:
                    yield f"event: message\ndata: __INTERRUPT__\n\n"
                    
                # 🌟 THE FIX 2: Smarter Title Trigger
                # Look up the current title. If it still ends with "…", it hasn't been renamed yet!
                # 🌟 THE FIX 2: Smarter Title Trigger with Threading
                # 🌟 THE FIX 2: Bulletproof Title Trigger
                if final_ai_text:
                    threads = _read_threads_metadata()
                    
                    # Check the flag instead of the string format
                    if thread_id in threads and not threads[thread_id].get("is_renamed", False):
                        user_question = state.values.get("question", "User Request")
                        
                        # Fire and forget in a completely separate thread!
                        threading.Thread(
                            target=update_thread_title_background, 
                            args=(thread_id, user_question, final_ai_text)
                        ).start()
        except Exception as e:
            logger.error("❌ Stream Exception caught!")
            traceback.print_exc()
            yield f"event: error\ndata: An internal error occurred.\n\n"
            
        finally:
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/chat/history")
async def get_history(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
        graph = build_graph(all_mcp_tools=[], checkpointer=checkpointer, ltm_store=ltm_store)
        state = await graph.aget_state({"configurable": {"thread_id": thread_id}})

        if not state or not state.values:
            return {"messages": [], "is_paused": False}

        is_paused = len(state.next) > 0

        formatted_msgs = []
        for msg in state.values.get("messages", []):
            if getattr(msg, 'type', '') in ['human', 'ai'] and getattr(msg, 'content', ''):
                if msg.type == 'ai' and getattr(msg, 'tool_calls', None):
                    continue

                if msg.type == 'ai' and getattr(msg, 'name', '') == 'internal_worker':
                    formatted_msgs.append({"role": "thought", "content": msg.content})
                else:
                    formatted_msgs.append({"role": "user" if msg.type == 'human' else "assistant", "content": msg.content})

        return {"messages": formatted_msgs, "is_paused": is_paused}

@app.get("/chat/threads", response_model=list[ThreadSummary])
async def list_threads(user_id: str):
    user_threads = [
        ThreadSummary(thread_id=tid, title=meta["title"], created_at=meta["created_at"])
        for tid, meta in _read_threads_metadata().items()
        if meta["user_id"] == user_id
    ]
    user_threads.sort(key=lambda t: t["created_at"], reverse=True)
    return user_threads

def register_thread(thread_id: str, user_id: str, first_message: str):
    """Called the first time a thread_id is used so it shows up in the sidebar."""
    threads = _read_threads_metadata()
    if thread_id not in threads:
        threads[thread_id] = {
            "user_id": user_id,
            "title": (first_message[:40] + "…") if len(first_message) > 40 else first_message,
            "created_at": datetime.utcnow().isoformat(),
            "is_renamed": False  # 🌟 NEW: Reliable tracking flag
        }
        _write_threads_metadata(threads)

@app.post("/admin/upload", response_model=UploadResponse)
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    user = USERS_DB.get(user_id)
    if not user or user["role"] != "ADMIN":
        raise HTTPException(
            status_code=403, detail="Only ADMIN users can upload documents"
        )

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_file(dest_path, original_filename=file.filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion failed for file '%s'", file.filename)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    docs = _read_docs_metadata()
    docs.append(
        {
            "filename": file.filename,
            "uploaded_by": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
    )
    _write_docs_metadata(docs)

    logger.info(f"📄 [UPLOAD] '{file.filename}' uploaded by admin '{user_id}'")
    return UploadResponse(
        filename=file.filename,
        status="stored",
        chunks_ingested=result["chunks_created"],
    )

@app.get("/admin/documents", response_model=list[DocumentInfo])
async def list_documents():
    docs = _read_docs_metadata()
    return sorted(docs, key=lambda d: d["uploaded_at"], reverse=True)