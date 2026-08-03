import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import traceback
from contextlib import asynccontextmanager
from datetime import datetime
from typing import TypedDict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from pydantic import BaseModel
from rag.ingestion.pipeline import ingest_file
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

# Import project modules
from main import build_graph
from Utils.logger import get_logger
from Config.llm_config import fast_llm

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

# =============================================================================
# Configure OpenTelemetry to export spans to the Phoenix server
# =============================================================================
endpoint = "http://127.0.0.1:6006/v1/traces"
tracer_provider = trace_sdk.TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))
trace_api.set_tracer_provider(tracer_provider)

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

app = FastAPI(title="Template Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Initialize Limiter (uses the user's IP address by default)
# =============================================================================
limiter = Limiter(key_func=get_remote_address)

# Inside your FastAPI app initialization:

# Register the limiter and exception handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# =============================================================================
# FILE PATHS & METADATA DATABASE SETUP
# =============================================================================
DATA_DIR = "data"
UPLOAD_DIR = "uploaded_docs"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Generic names for template reuse
APP_METADATA_DB = os.path.join(DATA_DIR, "app_metadata.db")
CHECKPOINTS_DB = os.path.join(DATA_DIR, "checkpoints.db")

def init_metadata_db():
    """Initializes the SQLite database for storing threads and document metadata."""
    with sqlite3.connect(APP_METADATA_DB) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                user_id TEXT,
                title TEXT,
                created_at TEXT,
                is_renamed BOOLEAN
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                filename TEXT PRIMARY KEY,
                uploaded_by TEXT,
                uploaded_at TEXT
            )
        """)

init_metadata_db()

USERS_DB: dict[str, dict] = {
    "admin1": {"password": "admin123", "role": "ADMIN"},
    "cust1":  {"password": "cust123",  "role": "CUSTOMER"},
    "rathan": {"password": "1234",  "role": "CUSTOMER"},
}

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
# SQLITE HELPER FUNCTIONS
# =============================================================================
def register_thread(thread_id: str, user_id: str, first_message: str):
    """Called the first time a thread_id is used."""
    with sqlite3.connect(APP_METADATA_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM threads WHERE thread_id = ?", (thread_id,))
        if not cursor.fetchone():
            title = (first_message[:40] + "…") if len(first_message) > 40 else first_message
            created_at = datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO threads (thread_id, user_id, title, created_at, is_renamed) VALUES (?, ?, ?, ?, ?)",
                (thread_id, user_id, title, created_at, False)
            )

def check_thread_needs_rename(thread_id: str) -> bool:
    """Checks if the thread is still using its default title."""
    with sqlite3.connect(APP_METADATA_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT is_renamed FROM threads WHERE thread_id = ?", (thread_id,))
        row = cursor.fetchone()
        return row is not None and not row[0]

def update_thread_title_background(thread_id: str, user_msg: str, ai_msg: str):
    """Runs in a completely separate thread to avoid FastAPI connection closures."""
    try:
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
            
        with sqlite3.connect(APP_METADATA_DB) as conn:
            conn.execute("UPDATE threads SET title = ?, is_renamed = 1 WHERE thread_id = ?", (new_title, thread_id))
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
@limiter.limit("10/minute")
async def chat_stream(
    request: Request,
    thread_id: str = Form(...),
    user_id: str = Form(...),
    message: str = Form(None),
    action: str = Form(None),
    image: UploadFile = File(None)
):
    logger.info(f"💬 [INCOMING REQUEST] Thread: '{thread_id}' | User: '{user_id}' | Message: '{message}' | Action: '{action}'")

    async def generate():
        image_path = None
        final_ai_text = ""     
        
        if image and image.filename:
            logger.info(f"📸 Image uploaded: {image.filename}")
            fd, temp_path = tempfile.mkstemp(suffix=f"_{image.filename}")
            with os.fdopen(fd, 'wb') as f:
                content = await image.read()
                f.write(content)
            image_path = temp_path

        try:
            # Updated Checkpointer path
            async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
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
                async for mode, payload in graph.astream(inputs, config=thread_config, stream_mode=["messages", "updates"]):
                    
                    if mode == "messages":
                        msg, metadata = payload
                        node_name = metadata.get("langgraph_node", "unknown")

                        if msg.content and isinstance(msg.content, str):
                            content_clean = msg.content.replace('\n', '\\n')

                            if node_name == "worker_agent":
                                yield f"event: thought\ndata: {content_clean}\n\n"

                    elif mode == "updates":
                        for node_name, state_update in payload.items():
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
                                    yield f"event: message\ndata: {content_clean}\n\n"

                # Check if paused at security gate
                state = await graph.aget_state(thread_config)
                if len(state.next) > 0:
                    yield f"event: message\ndata: __INTERRUPT__\n\n"
                    
                # Trigger Title Generation
                if final_ai_text and check_thread_needs_rename(thread_id):
                    user_question = state.values.get("question", "User Request")
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

import os
import tempfile
from fastapi import File, UploadFile, HTTPException
from Config.llm_config import client # Assuming your AzureOpenAI client is configured here

# 🌟 New STT Endpoint
@app.post("/chat/transcribe")
@limiter.limit("10/minute")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Receives an audio file from the frontend, sends it to Azure Whisper, 
    and returns the transcribed text.
    """
    logger.info(f"🎙️ Received audio file for transcription: {file.filename}")
    
    # Check if the file is an acceptable audio format
    allowed_types = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/m4a", "audio/webm"]
    if file.content_type not in allowed_types:
        # Fallback check based on extension if content_type is generic
        if not any(file.filename.endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".webm"]):
            raise HTTPException(status_code=400, detail="Invalid audio format.")

    # Whisper requires an actual file object with a filename, so we save it temporarily
    try:
        suffix = f".{file.filename.split('.')[-1]}" if "." in file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        # Open the temp file and send to Whisper
        with open(tmp_file_path, "rb") as audio_file:
            # Note: adjust the client call based on whether you use AzureOpenAI or OpenAI standard SDK
            transcription = client.audio.transcriptions.create(
                model="azure/genailab-maas-whisper", # Your specific MaaS model name
                file=audio_file,
                # Optional: prompt="Give it context about logistics, trucks, and cities to improve accuracy"
            )
            
        transcribed_text = transcription.text
        logger.info(f"✅ Transcription successful: '{transcribed_text}'")
        
        return {"text": transcribed_text}

    except Exception as e:
        logger.error(f"❌ Transcription failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio.")
        
    finally:
        # Always clean up the temporary file
        if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)



@app.get("/chat/history")
async def get_history(thread_id: str):
    # Updated Checkpointer path
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINTS_DB) as checkpointer:
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
    with sqlite3.connect(APP_METADATA_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id, title, created_at FROM threads WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        rows = cursor.fetchall()
        
    return [
        ThreadSummary(thread_id=row[0], title=row[1], created_at=row[2])
        for row in rows
    ]

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

    with sqlite3.connect(APP_METADATA_DB) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO documents (filename, uploaded_by, uploaded_at) VALUES (?, ?, ?)",
            (file.filename, user_id, datetime.utcnow().isoformat())
        )

    logger.info(f"📄 [UPLOAD] '{file.filename}' uploaded by admin '{user_id}'")
    return UploadResponse(
        filename=file.filename,
        status="stored",
        chunks_ingested=result["chunks_created"],
    )

@app.get("/admin/documents", response_model=list[DocumentInfo])
async def list_documents():
    with sqlite3.connect(APP_METADATA_DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, uploaded_by, uploaded_at FROM documents ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        
    return [
        DocumentInfo(filename=row[0], uploaded_by=row[1], uploaded_at=row[2])
        for row in rows
    ]


# Add to ENDPOINTS section in server.py
@app.post("/admin/scan_disruptions")
@limiter.limit("2/minute")
async def scan_disruptions():
    """
    Scans all hubs for real-time news using MCP, evaluates risk via LLM, 
    and automatically flags identified disruptions into the DB.
    """
    logger.info("🔍 [GLOBAL RISK SCAN] Initiating automated risk scan across hubs...")
    try:
        # 1. Locate loaded MCP tools from lifespan session
        news_tool = next((t for t in mcp_tools if t.name == "fetch_active_hub_news"), None)
        flag_tool = next((t for t in mcp_tools if t.name == "flag_disruptions"), None)

        if not news_tool or not flag_tool:
            raise HTTPException(status_code=503, detail="MCP tools not connected or unavailable.")

        # 2. Call news tool (passing empty list triggers automatic DB city lookup)
        logger.info("📡 Fetching news across all registered logistics locations...")
        news_data = await news_tool.ainvoke({"active_cities": []})

        # 3. Pass news to LLM to evaluate for critical hazards
        prompt = PromptTemplate.from_template("""
        You are a Supply Chain Risk Evaluator. Analyze the following news items collected from logistics hubs:
        {news_data}

        Identify any REAL, SEVERE physical logistics disruptions (floods, highway closures, bridge collapses, strikes, major accidents).
        Ignore generic weather or normal traffic reports.

        Output strictly a JSON list of identified disruptions in this format:
        [
            {{
                "location_name": "CityName",
                "hazard_type": "Flood/Closure/Strike",
                "hazard_summary": "Short concise summary of the issue"
            }}
        ]
        If no disruptions are found, output an empty array: []
        """)

        chain = prompt | fast_llm
        llm_res = await chain.ainvoke({"news_data": json.dumps(news_data)})

        # Clean JSON markdown formatting if present
        raw_content = llm_res.content.strip()
        if raw_content.startswith("```"):
            lines = raw_content.splitlines()
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].startswith("```"): lines = lines[:-1]
            raw_content = "\n".join(lines).strip()

        flagged_items = json.loads(raw_content)

        # 4. If hazards were detected, execute flag_disruptions tool
        if flagged_items and isinstance(flagged_items, list) and len(flagged_items) > 0:
            logger.info(f"🚨 [RISK SCAN] Disruption detected! Flagging {len(flagged_items)} items...")
            flag_res = await flag_tool.ainvoke({"disruptions": flagged_items})
            return {
                "status": "success", 
                "flagged_count": len(flagged_items), 
                "details": flagged_items,
                "result": flag_res
            }

        logger.info("✅ [RISK SCAN] Completed. No hazards detected.")
        return {"status": "success", "flagged_count": 0, "message": "No new disruptions detected."}

    except Exception as e:
        logger.error(f"❌ Failed global risk scan: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))