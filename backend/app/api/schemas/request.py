from pydantic import BaseModel
from typing import Optional,Dict


class AnalyzeCaseRequest(BaseModel):
    raw_text_input: str
    image_path: Optional[str] = None          # server-side path (legacy / testing)
    image_base64: Optional[str] = None         # base64-encoded image sent from the browser
    follow_up_answers: Optional[Dict[str, str]] = {}