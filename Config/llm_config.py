import os
import httpx
import ssl
import warnings
from urllib3.exceptions import InsecureRequestWarning
from langchain_openai import AzureOpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

from cache_config.semantic_cache import LocalSemanticCache

load_dotenv(override=True)

# Retrieve the key from environment variables
API_KEY = os.getenv("TCS_GENAI_API_KEY")

if not API_KEY:
    raise ValueError("TCS_GENAI_API_KEY not found! Ensure your .env file is set up correctly.")

# ==========================================
# 1. LAB ENVIRONMENT SSL BYPASS
# ==========================================
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['REQUESTS_CA_BUNDLE'] = ''
os.environ['CURL_CA_BUNDLE'] = ''
warnings.simplefilter('ignore', InsecureRequestWarning)

# 🌟 Single robust sync client with SSL verification disabled & 120s timeout
client = httpx.Client(verify=False, timeout=120.0)

# ==========================================
# 2. CONFIGURATION & KEYS
# ==========================================
BASE_URL = "https://genailab.tcs.in"

# ==========================================
# 3. LLM INITIALIZATIONS
# ==========================================

# PRIMARY LLM: Core agent tasks
primary_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.2
)

# FAST LLM: For Routing, Classification, and Summarization
fast_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure/genailab-maas-gpt-4o-mini", 
    api_key=API_KEY,
    http_client=client,
    temperature=0
)

# REASONING LLM: Specifically for your Orchestrator node
reasoning_llm = ChatOpenAI(
    base_url=BASE_URL,
    model="azure_ai/genailab-maas-DeepSeek-R1", 
    api_key=API_KEY,
    http_client=client,
    temperature=0.1
)



stt_client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    http_client=client # This safely injects your custom SSL-bypassing httpx client
)

# os.environ["TIKTOKEN_CACHE_DIR"] = "C:\\Users\\GenAIBLRTKCUSR20\\Documents\\Team19\BankingApp\\TiktokenCache"
# EMBEDDING MODEL: For your RAG/VectorDB implementation
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=BASE_URL,
    azure_deployment="azure/genailab-maas-text-embedding-3-large",
    api_key=API_KEY,
    openai_api_version="2023-05-15",
    http_client=client
)