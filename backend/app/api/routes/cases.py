from fastapi import APIRouter, HTTPException
import base64
import os
import tempfile

from app.api.schemas.request import AnalyzeCaseRequest
from app.api.schemas.response import AnalyzeCaseResponse
from app.state.factory import create_initial_state
from app.crew.runner import run_case

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeCaseResponse)
def analyze_case(payload: AnalyzeCaseRequest) -> AnalyzeCaseResponse:
    """
    Submit a pet health case for multi-agent analysis.

    Accepts either a server-side image_path (testing) or a browser-uploaded
    image_base64 string. When base64 is provided it is written to a temp file
    so the Image Agent can read it via its normal file-path interface.
    """
    # ── Resolve image path ────────────────────────────────────────────────────
    image_path = payload.image_path
    _tmp_file  = None   # keep reference so we can delete after the run

    if payload.image_base64 and not image_path:
        try:
            # Strip optional data-URI prefix (e.g. "data:image/jpeg;base64,...")
            b64_data = payload.image_base64
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]

            img_bytes = base64.b64decode(b64_data)

            # Write to a named temp file that survives until we delete it
            suffix = ".jpg"
            _tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            _tmp_file.write(img_bytes)
            _tmp_file.flush()
            _tmp_file.close()
            image_path = _tmp_file.name
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image_base64: {exc}") from exc

    # ── Build initial state ───────────────────────────────────────────────────
    state = create_initial_state(
        raw_text_input=payload.raw_text_input,
        image_path=image_path,
    )
    state["follow_up_answers"] = payload.follow_up_answers or {}

    try:
        state = run_case(state)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        # Clean up temp file regardless of success/failure
        if _tmp_file and os.path.exists(_tmp_file.name):
            os.unlink(_tmp_file.name)

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
        pet_profile=state.get("pet_profile", {}),
        extracted_symptoms=state.get("extracted_symptoms", []),
    )

