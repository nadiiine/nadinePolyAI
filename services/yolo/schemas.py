from pydantic import BaseModel
from typing import List


class YOLOResponse(BaseModel):
    prediction_uid: str
    detection_count: int
    labels: List[str]
    time_took: float