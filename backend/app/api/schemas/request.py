from pydantic import BaseModel
from typing import Optional,Dict


class AnalyzeCaseRequest(BaseModel):
    raw_text_input: str
    image_path: Optional[str] = None
    
    follow_up_answers: Optional[Dict[str, str]] = {}