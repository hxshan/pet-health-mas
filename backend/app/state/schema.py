from typing import Any, Dict, List, Optional, TypedDict


class PetCaseState(TypedDict, total=False):
    case_id: str
    raw_text_input: str

    pet_profile: Dict[str, Any]
    extracted_symptoms: List[str]
    structured_case: Dict[str, Any]

    image_available: bool
    image_path: Optional[str]

    intake_status: str
    follow_up_questions: List[str]
    follow_up_answers: Dict[str, Any]

    symptom_assessment: Dict[str, Any]
    image_assessment: Dict[str, Any]

    needs_more_info: bool
    uncertainty_flags: List[str]
    possible_out_of_scope: bool

    triage_result: Dict[str, Any]
    final_report: Dict[str, Any]

    trace_events: List[Dict[str, Any]]