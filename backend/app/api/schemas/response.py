from pydantic import BaseModel
from typing import Any, Dict, List, Optional


class AnalyzeCaseResponse(BaseModel):
    case_id: str
    status: str
    urgency_level: Optional[str] = None
    recommendation: Optional[str] = None
    uncertainty_status: Optional[str] = None
    follow_up_questions: List[str] = []
    final_report: Dict[str, Any] = {}
    symptom_assessment: Dict[str, Any] = {}
    image_assessment: Dict[str, Any] = {}
    message: str = ""
