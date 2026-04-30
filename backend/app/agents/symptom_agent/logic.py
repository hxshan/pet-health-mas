"""
Symptom Assessment Agent — core logic
======================================
Pure Python. No FastAPI, no CrewAI.

Entry point:
    run_symptom_assessment(case: dict) -> dict

Flow:
    1. check_input_sufficiency()   — validate required fields + at least 1 symptom
    2. build_model_input()         — map structured case → XGBoost input dict
    3. predict()                   — run XGBoost model   (predictor.py)
    4. analyse()                   — evaluate uncertainty (confidence.py)
    5. build_response()            — assemble final structured output
"""
from typing import Any, Dict, List

from app.tools.symptom_tools.predictor  import predict
from app.tools.symptom_tools.confidence import analyse

# ── Symptom flag fields the model understands ──────────────────────────────────
SYMPTOM_FLAGS: List[str] = [
    "vomiting", "diarrhea", "fever", "lethargy", "loss_appetite",
    "dehydration", "itching", "red_skin", "hair_loss", "skin_lesions",
    "wounds", "dark_urine", "pale_gums", "pale_eyelids", "tick_exposure",
    "pain_urinating", "frequent_urination", "blood_in_urine",
]

# Required pet profile fields
REQUIRED_PROFILE_FIELDS: List[str] = ["species", "age_years", "weight_kg"]


# ── Step 1: sufficiency check ──────────────────────────────────────────────────

def check_input_sufficiency(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that the case has enough data to run the symptom model.

    Checks:
      - Required profile fields: species, age_years, weight_kg
      - At least one symptom flag set to 1

    Args:
        case: Structured case dict from Agent 1 (Intake Agent).

    Returns:
        {
            "sufficient":     bool,
            "missing_fields": list[str]   # empty when sufficient
        }
    """
    missing: List[str] = []

    for field in REQUIRED_PROFILE_FIELDS:
        if not case.get(field):
            missing.append(field)

    has_symptom = any(case.get(flag, 0) == 1 for flag in SYMPTOM_FLAGS)
    if not has_symptom:
        missing.append("at_least_one_symptom_flag")

    return {
        "sufficient":     len(missing) == 0,
        "missing_fields": missing,
    }


# ── Step 2: build model input ──────────────────────────────────────────────────

def build_model_input(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalise fields from the structured case into a dict
    the XGBoost model accepts.

    Profile fields are required; symptom flags default to 0 if absent.

    Args:
        case: Structured case dict from Agent 1.

    Returns:
        Dict ready to pass into predict().
    """
    model_input: Dict[str, Any] = {
        # Profile (required)
        "species":   case.get("species", "unknown"),
        "breed":     case.get("breed", "unknown"),
        "sex":       case.get("sex", "unknown"),
        "neutered":  case.get("neutered", "unknown"),
        "age_years": float(case.get("age_years", 0)),
        "weight_kg": float(case.get("weight_kg", 0)),
        "vaccinated": int(case.get("vaccinated", 0)),

        # Visit history
        "num_previous_visits":   int(case.get("num_previous_visits", 0)),
        "prev_diagnosis_class":  int(case.get("prev_diagnosis_class", -1)),
        "days_since_last_visit": int(case.get("days_since_last_visit", 0)),
        "chronic_flag":          int(case.get("chronic_flag", 0)),
    }

    # Symptom flags — all default to 0
    for flag in SYMPTOM_FLAGS:
        model_input[flag] = int(case.get(flag, 0))

    return model_input


# ── Step 5: interpret result in plain language ─────────────────────────────────

def _build_interpretation(
    top_prediction: str,
    confidence: float,
    uncertainty_flag: bool,
    uncertainty_reason: str,
) -> str:
    """
    Produce a short, plain-language interpretation for the Triage Agent.
    Never claims a definitive diagnosis.
    """
    if uncertainty_flag:
        return (
            f"The symptom pattern shows a possible match for '{top_prediction}' "
            f"within supported conditions, but the result is unreliable: {uncertainty_reason}. "
            "Veterinary consultation recommended."
        )
    return (
        f"The symptom pattern is consistent with '{top_prediction}' "
        f"within supported conditions (confidence: {confidence:.0%}). "
        "This is a preliminary assessment only."
    )


# ── Main entry point ───────────────────────────────────────────────────────────

def run_symptom_assessment(case: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full Symptom Assessment Agent logic.

    Args:
        case: Structured case dict produced by Agent 1 (Intake Agent).
              Must include pet profile fields and binary symptom flags.

    Returns:
        Structured assessment dict:
        {
            "assessment_status":    "completed" | "needs_more_info" | "error",
            "top_prediction":       str,
            "confidence":           float,
            "alternatives":         [{"condition": str, "confidence": float}],
            "uncertainty_flag":     bool,
            "uncertainty_reason":   str,
            "top_gap":              float,
            "possible_out_of_scope": bool,
            "needs_more_info":      bool,
            "missing_fields":       list[str],
            "local_interpretation": str,
        }
    """
    # Step 1 — sufficiency check
    sufficiency = check_input_sufficiency(case)
    if not sufficiency["sufficient"]:
        return {
            "assessment_status":     "needs_more_info",
            "top_prediction":        "",
            "confidence":            0.0,
            "alternatives":          [],
            "uncertainty_flag":      True,
            "uncertainty_reason":    "Insufficient input data.",
            "top_gap":               0.0,
            "possible_out_of_scope": False,
            "needs_more_info":       True,
            "missing_fields":        sufficiency["missing_fields"],
            "local_interpretation":  (
                "Cannot assess symptoms — required fields are missing: "
                + ", ".join(sufficiency["missing_fields"])
            ),
        }

    # Step 2 — build model input
    model_input = build_model_input(case)

    # Step 3 — run predictor
    try:
        prediction = predict(model_input)
    except RuntimeError as exc:
        return {
            "assessment_status":     "error",
            "top_prediction":        "",
            "confidence":            0.0,
            "alternatives":          [],
            "uncertainty_flag":      True,
            "uncertainty_reason":    str(exc),
            "top_gap":               0.0,
            "possible_out_of_scope": False,
            "needs_more_info":       False,
            "missing_fields":        [],
            "local_interpretation":  "Model inference failed. Veterinary consultation recommended.",
        }

    # Step 4 — confidence / uncertainty analysis
    analysis = analyse(prediction)

    # Step 5 — assemble response
    interpretation = _build_interpretation(
        top_prediction     = prediction.top_label,
        confidence         = prediction.top_confidence,
        uncertainty_flag   = analysis["uncertainty_flag"],
        uncertainty_reason = analysis["uncertainty_reason"],
    )

    return {
        "assessment_status":     "completed",
        "top_prediction":        prediction.top_label,
        "confidence":            prediction.top_confidence,
        "alternatives":          prediction.top3[1:],   # top3 minus winner
        "uncertainty_flag":      analysis["uncertainty_flag"],
        "uncertainty_reason":    analysis["uncertainty_reason"],
        "top_gap":               analysis["top_gap"],
        "possible_out_of_scope": False,   # Agent 2 leaves this False; Triage Agent may override
        "needs_more_info":       False,
        "missing_fields":        [],
        "local_interpretation":  interpretation,
    }
