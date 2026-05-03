from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class AnalyzeCaseResponse(BaseModel):
    case_id: str
    status: str
    urgency_level: Optional[str] = None
    recommendation: Optional[str] = None
    uncertainty_status: Optional[Union[bool, str]] = None
    follow_up_questions: List[str] = []
    final_report: Optional[Union[str, Dict[str, Any]]] = None
    symptom_assessment: Optional[Dict[str, Any]] = {}
    image_assessment: Optional[Dict[str, Any]] = {}
    # Live case-building fields (populated after every Agent 1 run)
    pet_profile: Optional[Dict[str, Any]] = {}
    extracted_symptoms: Optional[List[str]] = []
    triage_result: Optional[Dict[str, Any]] = {}