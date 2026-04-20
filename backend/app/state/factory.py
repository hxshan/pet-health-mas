import uuid
from app.state.schema import PetCaseState


def create_initial_state(raw_text_input: str, image_path: str | None = None) -> PetCaseState:
    return {
        "case_id": str(uuid.uuid4()),
        "raw_text_input": raw_text_input,
        "pet_profile": {},
        "extracted_symptoms": [],
        "structured_case": {},
        "image_available": image_path is not None,
        "image_path": image_path,
        "intake_status": "pending",
        "follow_up_questions": [],
        "follow_up_answers": {},
        "symptom_assessment": {},
        "image_assessment": {},
        "needs_more_info": False,
        "uncertainty_flags": [],
        "possible_out_of_scope": False,
        "triage_result": {},
        "final_report": {},
        "trace_events": [],
    }