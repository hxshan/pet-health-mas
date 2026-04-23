# """
# Pipeline entry point for Pet Health MAS.

# Call `run_case(state)` from the API route or any other entry point.
# """
# """
# Pipeline entry point for Pet Health MAS.
# """
import json
import logging
from typing import Any, Dict

from app.crew.crew_setup import build_crew
from app.agents.intake_agent.agent import run as intake_run

logger = logging.getLogger(__name__)


def run_case(state: Dict[str, Any]) -> Dict[str, Any]:

    raw_text = state.get("raw_text_input", "")
    image_path = state.get("image_path")
    image_available = bool(state.get("image_available", image_path is not None))

    logger.info("Starting pet health pipeline. case_id=%s", state.get("case_id"))

    
    state = intake_run(state)

   
    intake_followups = state.get("follow_up_questions", [])

   
    raw_text = state.get("raw_text_input", "")
    image_available = bool(state.get("image_available", image_path is not None))

   
    crew = build_crew(image_available=image_available)

    crew_inputs = {
        "raw_text_input": raw_text,
        "image_path": image_path or "",
        "image_available": image_available,
    }

    try:
        crew_result = crew.kickoff(inputs=crew_inputs)

        raw_output = str(crew_result)

        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            parsed = {"raw_output": raw_output}

        state["triage_result"] = parsed
        state["final_report"] = parsed.get("final_report", {})

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)

        state["triage_result"] = {"error": str(exc)}
        state["final_report"] = {
            "summary": "Pipeline error — veterinary consultation recommended.",
            "disclaimer": "An internal error occurred. Please consult a vet.",
        }

    
    if not state.get("follow_up_questions"):
        state["follow_up_questions"] = intake_followups

    return state
