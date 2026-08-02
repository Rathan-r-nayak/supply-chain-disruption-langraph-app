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
        
        # Save uploaded image to temp file for Vision Node
        if image and image.filename:
            logger.info(f"📸 Image uploaded: {image.filename}")
            fd, temp_path = tempfile.mkstemp(suffix=f"_{image.filename}")
            with os.fdopen(fd, 'wb') as f:
                content = await image.read()
                f.write(content)
            image_path = temp_path

        try:
            async with AsyncSqliteSaver.from_conn_string("banking_checkpoints.db") as checkpointer:
                # 🌟 Now mcp_tools will actually contain your server's tools!
                graph = build_graph(all_mcp_tools=mcp_tools, checkpointer=checkpointer, ltm_store=ltm_store)
                thread_config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}

                if action == "approve":
                    inputs = None
                elif action == "reject":
                    cancellation_msg = "Transaction cancelled by user."
                    logger.info(f"🛑 [STREAM END] Thread: '{thread_id}' | Output: {cancellation_msg}")
                    yield f"data: {cancellation_msg}\n\n"
                    return
                else:
                    inputs = {
                        "question": message or "",
                        "worker_responses": [],
                        "messages": [HumanMessage(content=message)] if message else [],
                        "image_path": image_path  # Pass the image path into State
                    }
                    if message:
                        register_thread(thread_id, user_id, message)

                # Stream the messages normally
                async for msg, metadata in graph.astream(inputs, config=thread_config, stream_mode="messages"):
                    node_name = metadata.get("langgraph_node")

                    if msg.content and isinstance(msg.content, str):
                        content_clean = msg.content.replace('\n', '\\n')

                        if node_name == "worker_agent":
                            yield f"event: thought\ndata: {content_clean}\n\n"
                        elif node_name in ["aggregator", "output_guardrail", "triage_router"]:
                            yield f"event: message\ndata: {content_clean}\n\n"

                # Check if paused at security gate
                state = await graph.aget_state(thread_config)
                if len(state.next) > 0:
                    yield f"event: message\ndata: __INTERRUPT__\n\n"

        except Exception as e:
            logger.error("❌ Stream Exception caught!")
            traceback.print_exc()
            yield f"event: error\ndata: An internal error occurred.\n\n"
            
        finally:
            # Clean up the image file after generation completes
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
                logger.info(f"🗑️ Cleaned up temp image file: {image_path}")

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
        }
        _write_threads_metadata(threads)

@app.post("/admin/upload", response_model=UploadResponse)
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
):
    user = USERS_DB.get(user_id)
    if not user or user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Only ADMIN users can upload documents")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Note: RAG Pipeline ingestion logic should hook in here in the future
    
    docs = _read_docs_metadata()
    docs.append({
        "filename": file.filename,
        "uploaded_by": user_id,
        "uploaded_at": datetime.utcnow().isoformat(),
    })
    _write_docs_metadata(docs)

    logger.info(f"📄 [UPLOAD] '{file.filename}' uploaded by admin '{user_id}'")
    return UploadResponse(filename=file.filename, status="stored", chunks_ingested=None)

@app.get("/admin/documents", response_model=list[DocumentInfo])
async def list_documents():
    docs = _read_docs_metadata()
    return sorted(docs, key=lambda d: d["uploaded_at"], reverse=True)