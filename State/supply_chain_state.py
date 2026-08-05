import operator
from typing import Annotated, Literal, Optional, Sequence, TypedDict

from langgraph.graph import add_messages
from langgraph.store.memory import InMemoryStore
from langchain_core.messages import BaseMessage
from pydantic import Field

from Schema.task_schema import Task

def merge_lists(left: list | None, right: list | None) -> list:
    if right == []:
        return []
    return (left or []) + (right or [])


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
    loop_count: int
    is_cache_hit: bool

    is_sufficient: bool
    relevance_score: str
    knowledge_retries: int

    next_best_actions: list[str]
    chart_payload: dict

    conversation_summary: str
    generation: str
    

class WorkerState(TypedDict):
    task: Task
    messages: Annotated[list, add_messages]
    worker_responses: list[str]

    executed_tools: list[str]

memory_store = InMemoryStore()