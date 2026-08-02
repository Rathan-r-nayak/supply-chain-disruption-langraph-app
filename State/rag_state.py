from typing import TypedDict, List, Dict, Any, Union
from langchain_core.messages import BaseMessage


class VectorFact(TypedDict):
    content: str
    source: str
    score: float


class HybridDocuments(TypedDict):
    vector_facts: List[VectorFact]
    graph_facts_used: List[str]


class RagState(TypedDict):
    question: str
    messages: List[BaseMessage]
    # 🌟 Updated to accept structured hybrid context (or Union if web search returns a raw string)
    documents: HybridDocuments  # Or Union[HybridDocuments, str]
    relevance_score: str
    knowledge_retries: int
    generation: str