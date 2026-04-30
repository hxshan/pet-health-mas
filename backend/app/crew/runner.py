# app/crew/runner.py

import copy
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.intake_agent.agent import run as intake_run, generate_targeted_followup
from app.agents.symptom_agent.logic import run_symptom_assessment
from app.agents.image_agent.agent import run as image_agent_run
from app.crew.crew_setup import build_crew

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float(val, default=0.0):
    if val is None:
        return default
    try:
        return float(str(val).split()[0])
    except (ValueError, IndexError):
        return default


def _run_symptom_agent(state: dict) -> dict:
    """Build case_data and call Agent 2. Returns updated symptom_assessment."""
    raw_case = state.get("structured_case") or {}
    profile  = raw_case.get("pet_profile", {})
    symptoms = raw_case.get("symptoms", state.get("extracted_symptoms", []))

    case_data = {
        "species":   profile.get("species", "unknown"),
        "breed":     profile.get("breed",   "unknown"),
        "sex":       profile.get("sex",     "unknown"),
        "neutered":  profile.get("neutered", "unknown"),
        "age_years": _to_float(profile.get("age")),
        "weight_kg": _to_float(profile.get("weight")),
    }

    symptom_map = {
        "vomiting":           "vomiting",
        "diarrhea":           "diarrhea",
        "fever":              "fever",
        "lethargy":           "lethargy",
        "loss_of_appetite":   "loss_appetite",
        "loss_appetite":      "loss_appetite",
        "dehydration":        "dehydration",
        "itching":            "itching",
        "red_skin":           "red_skin",
        "hair_loss":          "hair_loss",
        "skin_lesions":       "skin_lesions",
        "wounds":             "wounds",
        "dark_urine":         "dark_urine",
        "pale_gums":          "pale_gums",
        "pale_eyelids":       "pale_eyelids",
        "tick_exposure":      "tick_exposure",
        "pain_urinating":     "pain_urinating",
        "frequent_urination": "frequent_urination",
        "blood_in_urine":     "blood_in_urine",
    }
    for s in symptoms:
        flag = symptom_map.get(s.lower().replace(" ", "_"))
        if flag:
            case_data[flag] = 1

    return run_symptom_assessment(case_data)


def _run_image_agent(state: dict) -> dict:
    """
    Run the Image Assessment Agent.
    Receives a shallow copy of state so the thread doesn't mutate shared state.
    Returns the image_assessment dict (or a stub when no image present).
    """
    # Image agent is already a no-op when image_available is False
    result_state = image_agent_run(copy.copy(state))
    return result_state.get("image_assessment", {})


# ── main entry point ──────────────────────────────────────────────────────────

def run_case(state: dict) -> dict:
    """
    Run the full multi-agent pipeline for one pet health case.

    Steps
    -----
    1. Agent 1  — Intake (LLM entity extraction)
    2. Early return if Agent 1 still needs more info
    3. Agent 2 + Image Agent run **in parallel** via ThreadPoolExecutor
       3a. If Agent 2 signals needs_more_info → hand back to Agent 1, return early
    4. Agents 3 & 4 — Triage via CrewAI (receives both assessments)
    """
    raw_text        = state.get("raw_text_input", "")
    image_path      = state.get("image_path")
    image_available = bool(state.get("image_available", image_path is not None))

    # ── Step 1: Agent 1 ───────────────────────────────────────────────────────
    state = intake_run(state)

    # ── Step 2: early exit if intake incomplete ───────────────────────────────
    if state.get("intake_status") != "complete":
        return state

    # ── Step 3: Agent 2 + Image Agent in parallel ─────────────────────────────
    symptom_result: dict = {}
    image_result:   dict = {}

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_symptom = pool.submit(_run_symptom_agent, state)
        fut_image   = pool.submit(_run_image_agent,   state)

        for fut in as_completed([fut_symptom, fut_image]):
            try:
                if fut is fut_symptom:
                    symptom_result = fut.result()
                    logger.info(
                        "Symptom assessment: status=%s top=%s conf=%.2f",
                        symptom_result.get("assessment_status"),
                        symptom_result.get("top_prediction"),
                        symptom_result.get("confidence", 0),
                    )
                else:
                    image_result = fut.result()
                    logger.info(
                        "Image assessment: validity=%s prediction=%s",
                        image_result.get("image_validity"),
                        image_result.get("image_prediction"),
                    )
            except Exception as exc:
                if fut is fut_symptom:
                    logger.warning("Symptom assessment failed: %s", exc)
                    symptom_result = {"error": str(exc), "assessment_status": "error"}
                else:
                    logger.warning("Image assessment failed: %s", exc)
                    image_result = {"error": str(exc), "image_validity": "error"}

    state["symptom_assessment"] = symptom_result
    state["image_assessment"]   = image_result

    # ── Step 3a: Agent 2 needs more info → route back through Agent 1 ─────────
    if symptom_result.get("assessment_status") == "needs_more_info":
        missing = symptom_result.get("missing_fields", [])
        logger.info("Agent 2 needs more info: %s — asking Agent 1 to follow up", missing)

        question = generate_targeted_followup(
            missing_fields  = missing,
            structured_case = state.get("structured_case", {}),
        )
        if not question:
            human    = [f.replace("_", " ") for f in missing]
            question = f"To continue the assessment, could you please provide: {', '.join(human)}?"

        state["follow_up_questions"]    = [question]
        state["intake_status"]          = "needs_more_info"
        state["awaiting_agent2_retry"]  = True
        return state

    # ── Step 4: Agents 3 & 4 — Triage via CrewAI ──────────────────────────────
    crew = build_crew(image_available=image_available)

    crew_inputs = {
        "raw_text_input":     raw_text,
        "image_path":         image_path or "",
        "image_available":    image_available,
        "symptom_assessment": json.dumps(symptom_result),
        "image_assessment":   json.dumps(image_result),
    }

    try:
        result = crew.kickoff(inputs=crew_inputs)
        try:
            parsed = json.loads(str(result))
        except json.JSONDecodeError:
            parsed = {"raw_output": str(result)}

        state["triage_result"] = parsed
        state["final_report"]  = parsed.get("final_report", {})

    except Exception as exc:
        logger.exception("Crew pipeline failed: %s", exc)
        state["triage_result"] = {"error": str(exc)}
        state["final_report"]  = {
            "summary":    "Pipeline error — veterinary consultation recommended.",
            "disclaimer": "An internal error occurred. Please consult a vet.",
        }

    return state
