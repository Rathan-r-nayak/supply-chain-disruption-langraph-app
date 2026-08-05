# Schema/task.py
from pydantic import BaseModel, Field
from typing import List, Literal, TypedDict

class Task(BaseModel):
    task_id: str = Field(description="A unique identifier for the task (e.g., 'task_1')")
    description: str = Field(description="Specific instruction for the worker")
    tool_type: Literal["safe", "sensitive", "rag"] = Field(description="The type of tools needed")
    assigned_worker: str = Field(description="The worker assigned to this task")


class OrchestratorPlan(BaseModel):
    is_workflow_complete: bool = Field(description="True if the user's request is fully answered")
    tasks: List[Task] = Field(default_factory=list, description="List of tasks for the workers to execute in parallel")
    final_answer: str = Field(default="", description="The final answer to the user if workflow is complete")