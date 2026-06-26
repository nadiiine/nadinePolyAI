from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class AgentResponse(BaseModel):
    response: str
    prediction_id: str
    annotated_image: Optional[str] = None
    agent_loop_time_s: float
    iterations: int
    tools_called: List[str]
    context_limit_exceeded: bool
    tokens_used: Dict[str, int]