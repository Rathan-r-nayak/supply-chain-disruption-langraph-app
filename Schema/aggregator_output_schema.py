from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ChartDataset(BaseModel):
    label: str
    data: List[float]

class ChartPayload(BaseModel):
    chart_type: str = Field(description="Type of chart: 'bar', 'line', or 'pie'")
    title: str = Field(description="Title of the chart")
    labels: List[str] = Field(description="X-axis labels or categories")
    datasets: List[ChartDataset]

class AggregatorOutput(BaseModel):
    final_answer: str = Field(description="The main Markdown response for the user, including citations.")
    is_chartable: bool = Field(description="True ONLY if the worker data contains distinct numeric comparisons, delays, volumes, or time-series data suitable for a chart.")
    chart_payload: Optional[ChartPayload] = Field(default=None, description="Chart configuration if is_chartable is True")
