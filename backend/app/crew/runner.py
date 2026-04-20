"""
Pipeline entry point for Pet Health MAS.

Call `run_case(state)` from the API route or any other entry point.
"""
import json
import logging
from typing import Any, Dict

from app.crew.crew_setup import build_crew

logger = logging.getLogger(__name__)


def run_case(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full multi-agent pipeline for one pet health case.

    Args:
        state: A PetCaseState-compatible dict (see app/state/schema.py).
               Must contain at least `raw_text_input`.

    Returns:
        The same state dict enriched with:
          - symptom_assessment
          - image_assessment  (if image was present)
          - triage_result
          - final_report
    """
    raw_text  = state.get("raw_text_input", "")
    image_path = state.get("image_path")
    image_available = bool(state.get("image_available", image_path is not None))

    logger.info("Starting pet health pipeline. case_id=%s", state.get("case_id"))

    crew = build_crew(image_available=image_available)

    crew_inputs = {
        "raw_text_input": raw_text,
        "image_path": image_path or "",
        "image_available": image_available,
    }

    try:
        crew_result = crew.kickoff(inputs=crew_inputs)

        # CrewAI returns the last task output as a string — try to parse JSON
        raw_output = str(crew_result)
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = {"raw_output": raw_output}

        state["triage_result"] = parsed
        state["final_report"]  = parsed.get("final_report", {})

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        state["triage_result"] = {"error": str(exc)}
        state["final_report"]  = {
            "summary": "Pipeline error — veterinary consultation recommended.",
            "disclaimer": "An internal error occurred. Please consult a vet.",
        }

    return state
