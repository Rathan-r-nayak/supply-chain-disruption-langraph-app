import os
import httpx
import ssl
import warnings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from urllib3.exceptions import InsecureRequestWarning
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI, OpenAI
from dotenv import load_dotenv
from langchain_groq import ChatGroq


load_dotenv(override=True)

API_KEY = os.getenv("TCS_GENAI_API_KEY")
if not API_KEY:
    raise ValueError("TCS_GENAI_API_KEY not found!")

# 1. SSL BYPASS
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
warnings.simplefilter('ignore', InsecureRequestWarning)

# Single robust client
client = httpx.Client(verify=False, timeout=120.0)
BASE_URL = "https://genailab.tcs.in"

# ==========================================
# 2. THE "MIXTURE OF MODELS" LLM TIERS
# ==========================================

# 🧠 TIER 1: THE BRAIN (Highest Reasoning, High Cost, Low Frequency)
# Used for: Orchestrator
brain_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="genailab-maas-gpt-5.2", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)
# atest_brain_llmlternative of gpt-5.2
test_brain_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)

# 🛠️ TIER 2: THE WORKER (Flawless Tool Calling, Medium Cost, Loop Frequency)
# Used for: worker_agent
tool_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4.1", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.0
)

# ✍️ TIER 3: THE SYNTHESIZER (Good Generation, Medium Cost, Low Frequency)
# Used for: Aggregator, RAG generate_node
synthesis_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="genailab-maas-gpt-5.4-mini", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.2
)

# 🚦 TIER 4: THE ROUTER (Fast Classification, Low Cost, High Frequency)
# Used for: triage_router, evaluate_node, rewrite_node, summarize_conversation
fast_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4.1-mini", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.0
)

# 🛡️ TIER 5: THE GUARD (Mechanical Binary Checks, Lowest Cost, Highest Frequency)
# Used for: input_guardrail, output_guardrail
nano_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4.1-nano", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.0
)

# 👁️ TIER 6: THE EYE (Multimodal Processing)
# Used for: vision_node
vision_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)

# 🗃️ TIER 7: EMBEDDINGS (Vector Math)
# Used for: retrieve_node, semantic_cache
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=BASE_URL,
    azure_deployment="azure/genailab-maas-text-embedding-3-large",
    api_key=API_KEY,
    openai_api_version="2023-05-15",
    http_client=client
)

stt_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=client # This safely injects your custom SSL-bypassing httpx client
)






# import os
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# # ==========================================
# # LOCAL OLLAMA CONFIGURATION (No API Key Needed)
# # ==========================================
# OLLAMA_BASE_URL = "http://localhost:11434/v1"
# DUMMY_API_KEY = "ollama"  # Ollama requires a non-empty string for key validation

# # 🧠 TIER 1: THE BRAIN (Orchestrator Node)
# brain_llm = ChatOpenAI(
#     base_url=OLLAMA_BASE_URL,
#     model="qwen-2.5.1-coder-it:latest",
#     api_key=DUMMY_API_KEY,
#     temperature=0.1
# )

# # 🛠️ TIER 2: THE WORKER (Tool Calling)
# tool_llm = ChatOpenAI(
#     base_url=OLLAMA_BASE_URL,
#     model="qwen-2.5.1-coder-it:latest",
#     api_key=DUMMY_API_KEY,
#     temperature=0.0
# )

# # ✍️ TIER 3: THE SYNTHESIZER (Aggregator & RAG Generation)
# synthesis_llm = ChatOpenAI(
#     base_url=OLLAMA_BASE_URL,
#     model="gemma-3-4b-it:latest",
#     api_key=DUMMY_API_KEY,
#     temperature=0.2
# )

# # 🚦 TIER 4: THE ROUTER (Triage & Classification)
# fast_llm = ChatOpenAI(
#     base_url=OLLAMA_BASE_URL,
#     model="llama-3.2-3b-it:latest",
#     api_key=DUMMY_API_KEY,
#     temperature=0.0
# )

# # 🛡️ TIER 5: THE GUARD (Input/Output Safety)
# nano_llm = ChatOpenAI(
#     base_url=OLLAMA_BASE_URL,
#     model="llama-3.2-3b-it:latest",
#     api_key=DUMMY_API_KEY,
#     temperature=0.0
# )

# # 👁️ TIER 6: VISION (Fallback to Gemma for local text testing)
# vision_llm = synthesis_llm

# # 🗃️ TIER 7: EMBEDDINGS (RAG Vector Store & Semantic Cache)
# embedding_model = OpenAIEmbeddings(
#     base_url=OLLAMA_BASE_URL,
#     model="gte-large:latest",
#     api_key=DUMMY_API_KEY
# )











openrouter_key = os.getenv("OPENROUTER_API_KEY")


# local testing
test_brain_llm = ChatGoogleGenerativeAI(model='models/gemini-2.5-flash', temperature=0)

reasoning_llm = ChatOllama(model="llama3.2", temperature=0)

nano_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

synthesis_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2)

fast_llm = ChatOpenAI(
    openai_api_key=openrouter_key,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openrouter/free", 
    temperature=0.1,
    default_headers={
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Smart Helpdesk Triage App"
    }
)

# reasoning_llm = ChatOpenAI(
#     model="gemma2:2b",
#     openai_api_key="ollama", # Placeholder string to pass LangChain initialization checks
#     openai_api_base="http://localhost:11434/v1", # Native Ollama local port mapping
#     temperature=0.3,
# )
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
