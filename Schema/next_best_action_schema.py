from pydantic import BaseModel, Field


class NextBestActionSchema(BaseModel):
    suggestions: list[str] = Field(
        description="2 to 3 concise, highly actionable follow-up prompts or actions relevant to the conversation.",
        max_items=3
    )