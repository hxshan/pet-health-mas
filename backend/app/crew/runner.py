# app/crew/runner.py

import copy
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.agents.intake_agent.agent import run as intake_run, generate_targeted_followup
from app.agents.symptom_agent.logic import run_symptom_assessment
from app.agents.image_agent.agent import run as image_agent_run
from app.agents.triage_agent.agent import run as triage_run

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_float(val, default=0.0):
    if val is None:
        return default
    # Extract the first number (int or decimal) from strings like "3 years", "4kg", "2.5"
    m = re.search(r'[\d]+(?:\.\d+)?', str(val))
    if m:
        try:
            return float(m.group())
        except ValueError:
            pass
    return default


def _run_symptom_agent(state: dict) -> dict:
    """Build case_data and call Agent 2. Returns updated symptom_assessment."""
    raw_case       = state.get("structured_case") or {}
    profile        = raw_case.get("pet_profile", {})
    prev_profile   = state.get("pet_profile", {})  # previously extracted (fallback)
    symptoms       = raw_case.get("symptoms", state.get("extracted_symptoms", []))

    def _get(key):
        """Return the first non-'unknown'/non-empty value from current or previous profile."""
        for p in (profile, prev_profile):
            v = p.get(key)
            if v and str(v).lower() not in ("unknown", "none", ""):
                return v
        return None

    case_data = {
        "species":   _get("species") or "unknown",
        "breed":     _get("breed")   or "unknown",
        "sex":       _get("sex")     or "unknown",
        "neutered":  _get("neutered") or "unknown",
        "age_years": _to_float(_get("age")),
        "weight_kg": _to_float(_get("weight")),
    }

    symptom_map = {
        "vomiting":           "vomiting",
        "diarrhea":           "diarrhea",
        "fever":              "fever",
        "lethargy":           "lethargy",
        "loss_of_appetite":   "loss_appetite",
        "loss_appetite":      "loss_appetite",
        "no_appetite":        "loss_appetite",
        "not_eating":         "loss_appetite",
        "dehydration":        "dehydration",
        "itching":            "itching",
        "scratching":         "itching",
        "red_skin":           "red_skin",
        "hair_loss":          "hair_loss",
        "skin_lesions":       "skin_lesions",
        "wounds":             "wounds",
        "wound":              "wounds",
        "dark_urine":         "dark_urine",
        "pale_gums":          "pale_gums",
        "pale_eyelids":       "pale_eyelids",
        "tick_exposure":      "tick_exposure",
        "pain_urinating":     "pain_urinating",
        "frequent_urination": "frequent_urination",
        "blood_in_urine":     "blood_in_urine",
        # extras the LLM commonly returns
        "cough":              "vomiting",   # no direct model flag; map to closest respiratory proxy
        "coughing":           "vomiting",
        "sneezing":           "vomiting",
        "breathing_difficulty": "vomiting",
        "tiredness":          "lethargy",
        "tired":              "lethargy",
        "red_patches":        "red_skin",
        "redness":            "red_skin",
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
    result_state = image_agent_run(copy.copy(state))
    return result_state.get("image_assessment", {})


def _run_triage(symptom_result: dict, image_result: dict, pet_profile: dict) -> dict:
    """
    Agent 4 — lightweight triage via a single direct Ollama call.
    Receives the already-computed Agent 2 + Image Agent outputs and
    returns urgency level, recommendation, and a final report.
    No CrewAI, no re-running of tools.
    """
    symptom_summary = (
        f"Top prediction: {symptom_result.get('top_prediction', 'unknown')} "
        f"({round((symptom_result.get('confidence') or 0) * 100)}% confidence). "
        f"Alternatives: {symptom_result.get('alternatives', [])}. "
        f"Uncertainty: {symptom_result.get('uncertainty_flag', False)}."
    ) if symptom_result.get("assessment_status") == "completed" else (
        f"Symptom assessment status: {symptom_result.get('assessment_status', 'unknown')}."
    )

    image_summary = "No image provided." if not image_result else (
        f"Image prediction: {image_result.get('image_prediction', 'unknown')} "
        f"({round((image_result.get('confidence') or 0) * 100)}% confidence). "
        f"Validity: {image_result.get('image_validity', 'unknown')}. "
        f"Uncertainty: {image_result.get('uncertainty_flag', False)}."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a veterinary triage specialist.\n"
                "You receive pre-computed symptom and image assessment results.\n"
                "Your job: synthesise them, assign an urgency level, and write a short recommendation.\n\n"
                "Urgency levels (pick exactly one): non-urgent | monitor | vet_soon | urgent\n\n"
                "Rules:\n"
                "- NEVER give a definitive diagnosis\n"
                "- Always recommend professional veterinary consultation\n"
                "- Be concise — max 2 sentences for recommendation\n"
                "- Return ONLY valid JSON, no markdown"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Pet profile: {json.dumps(pet_profile)}\n\n"
                f"Symptom assessment: {symptom_summary}\n\n"
                f"Image assessment: {image_summary}\n\n"
                "Return ONLY:\n"
                "{\n"
                '  "urgency_level": "",\n'
                '  "recommendation": "",\n'
                '  "uncertainty_status": "",\n'
                '  "final_report": {"summary": "", "disclaimer": "Always consult a qualified veterinarian."}\n'
                "}"
            ),
        },
    ]

    try:
        result = chat_json(messages)
        if isinstance(result, dict):
            return result
    except Exception as exc:
        logger.warning("Triage LLM call failed: %s", exc)

    # Fallback if LLM fails or returns non-JSON
    return {
        "urgency_level": "monitor",
        "recommendation": "Based on the assessment, please consult a veterinarian for proper evaluation.",
        "uncertainty_status": "low_confidence",
        "final_report": {
            "summary": f"Possible: {symptom_result.get('top_prediction', 'unknown')}. "
                       f"Please seek veterinary advice.",
            "disclaimer": "Always consult a qualified veterinarian.",
        },
    }


# ── main entry point ──────────────────────────────────────────────────────────

def run_case(state: dict) -> dict:
    """
    Run the multi-agent pipeline for one pet health case.

    Steps
    -----
    1. Agent 1  — Intake (LLM entity extraction + follow-up questions)
    2. Early return if Agent 1 still needs more info
    3. Agent 2 + Image Agent run in parallel via ThreadPoolExecutor
       3a. If Agent 2 signals needs_more_info → route back to Agent 1
    4. Return Agent 2 + Agent 3 results directly (Triage/Agent 4 not yet active)
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

    # ── Step 4: Triage skipped — return Agent 2 + Agent 3 results directly ───
    state = triage_run(state)
    logger.info(
        "Triage complete: urgency=%s uncertainty=%s",
        state.get("triage_result", {}).get("urgency_level"),
        state.get("triage_result", {}).get("uncertainty_status"),
    )

    return state

    return state
