import operator
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langgraph.graph import add_messages
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import BaseMessage

def merge_lists(left: list | None, right: list | None) -> list:
    if right == []:
        return []
    return (left or []) + (right or [])

class Task(TypedDict):
    task_id: str
    description: str
    assigned_worker: str
    tool_type: Literal["safe", "sensitive", "rag"]

class SupplyChainState(TypedDict, total=False):
    question: str
    messages: Annotated[Sequence[BaseMessage], add_messages]
    memories: str
    image_path: Optional[str]

    is_safe: bool
    requires_workflow: bool
    
    is_workflow_complete: bool 
    
    documents: list[dict]

    tasks: list[Task]
    worker_responses: Annotated[list[str], merge_lists]
    is_cache_hit: bool

    is_sufficient: bool
    relevance_score: str
    knowledge_retries: int

    generation: str

class WorkerState(TypedDict):
    task: Task
    messages: Annotated[list, add_messages]
    worker_responses: list[str]

memory_store = InMemoryStore()