from fastapi import APIRouter, HTTPException
from app.api.schemas.request import AnalyzeCaseRequest
from app.api.schemas.response import AnalyzeCaseResponse
from app.state.factory import create_initial_state
from app.crew.runner import run_case

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeCaseResponse)
def analyze_case(payload: AnalyzeCaseRequest) -> AnalyzeCaseResponse:
    """
    Submit a pet health case for multi-agent analysis.

    The pipeline runs synchronously (Ollama is local).
    For long-running cases consider moving to a background task.
    """
    state = create_initial_state(
        raw_text_input=payload.raw_text_input,
        image_path=payload.image_path,
    )

    try:
        state = run_case(state)
    except RuntimeError as exc:
        # Surface Ollama connectivity errors as 503
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    triage = state.get("triage_result", {})

    return AnalyzeCaseResponse(
        case_id=state["case_id"],
        status="complete",
        urgency_level=triage.get("urgency_level"),
        recommendation=triage.get("recommendation"),
        uncertainty_status=triage.get("uncertainty_status"),
        follow_up_questions=state.get("follow_up_questions", []),
        final_report=state.get("final_report", {}),
        symptom_assessment=state.get("symptom_assessment", {}),
        image_assessment=state.get("image_assessment", {}),
    )
