# app/crew/runner.py

from app.agents.intake_agent.agent import run as intake_run
from app.crew.crew_setup import build_crew
import json
from app.agents.symptom_agent.logic import run_symptom_assessment


def run_case(state):
    Returns:
        The same state dict enriched with:
          - symptom_assessment
          - image_assessment  (if image was present)
          - triage_result
          - final_report
    """
    raw_text   = state.get("raw_text_input", "")
    image_path = state.get("image_path")
    image_available = bool(state.get("image_available", image_path is not None))

    # 1️⃣ Run Agent 1 ONLY
    state = intake_run(state)

    # 2️⃣ STOP if not enough info
    if state.get("intake_status") != "complete":
        return state

    # 3️⃣ THEN run Agent 2/3/4
    crew = build_crew(image_available=False)
    # ------------------------------------------------------------------
    # Agent 2 — Symptom Assessment (called directly for reliability)
    # The XGBoost model + confidence analysis runs deterministically;
    # no LLM round-trip is needed for this step.
    # ------------------------------------------------------------------
    try:
        case_data = state.get("structured_case") or state
        symptom_result = run_symptom_assessment(case_data)
        state["symptom_assessment"] = symptom_result
        logger.info(
            "Symptom assessment complete. top=%s conf=%.2f",
            symptom_result.get("top_prediction"),
            symptom_result.get("confidence", 0),
        )
    except Exception as exc:
        logger.warning("Symptom assessment failed: %s", exc)
        state["symptom_assessment"] = {"error": str(exc), "assessment_status": "error"}

    # ------------------------------------------------------------------
    # CrewAI crew — remaining agents (intake stub, image, triage)
    # ------------------------------------------------------------------
    crew = build_crew(image_available=image_available)

    crew_inputs = {
        "raw_text_input": raw_text,
        "image_path": image_path or "",
        "image_available": image_available,
        # pass symptom result so triage agent can reference it
        "symptom_assessment": json.dumps(state["symptom_assessment"]),
    }

    try:
        result = crew.kickoff(inputs=state)
        parsed = json.loads(str(result))

        state["triage_result"] = parsed
        state["final_report"] = parsed.get("final_report", {})

    except Exception:
        state["final_report"] = {
            "summary": "Pipeline error — veterinary consultation recommended."
    except Exception as exc:
        logger.exception("Crew pipeline failed: %s", exc)
        state["triage_result"] = {"error": str(exc)}
        state["final_report"]  = {
            "summary": "Pipeline error — veterinary consultation recommended.",
            "disclaimer": "An internal error occurred. Please consult a vet.",
        }

    return state