from typing import Optional
from pydantic import BaseModel, Field


class PrimaryClassifierDecision(BaseModel):
    is_workflow_required: bool = Field(
        description="True if the user is asking about banking, accounts, transactions, or policies. False for greetings, pleasantries, or completely off-topic questions."
    )
    direct_response: Optional[str] = Field(
        description="If is_workflow_required is False, provide the conversational response or polite refusal here. If True, return None."
    )