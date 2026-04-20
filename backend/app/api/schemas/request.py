from pydantic import BaseModel
from typing import Optional


class AnalyzeCaseRequest(BaseModel):
    raw_text_input: str
    image_path: Optional[str] = None