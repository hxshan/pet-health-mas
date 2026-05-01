from __future__ import annotations
"""
Triage / Urgency Calculator Tool
===================================
Combines symptom and image assessment outputs and assigns an urgency level.

Expected output schema from _run():
{
  "urgency_level": str,        # "non-urgent" | "monitor" | "vet_soon" | "urgent"
  "recommendation": str,
  "uncertainty_status": str,   # "confident" | "uncertain" | "conflict" | "out_of_scope"
  "agreement_status": str,     # "agree" | "conflict" | "symptom_only" | "none"
  "reasoning": str
}
"""

import json
import logging
from typing import Any, Dict

from crewai.tools import BaseTool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Urgency rules (deterministic — no LLM dependency)
# ---------------------------------------------------------------------------

_URGENT_CONDITIONS: set[str] = {
    "poisoning", "toxicity", "anaphylaxis", "respiratory_distress",
    "bloat", "gastric_dilatation", "trauma", "severe_injury",
    "cardiac", "heart_failure", "seizure", "neurological",
    "haemobartonellosis", "ehrlichiosis", "babesiosis",
}

_VET_SOON_CONDITIONS: set[str] = {
    "parvovirus", "distemper", "pancreatitis", "urinary_obstruction",
    "diabetes", "kidney_disease", "liver_disease", "anaemia",
    "skin_infection", "mange", "ringworm", "leptospirosis",
    "tick_fever", "canine_tick_fever",
}

_CONFIDENCE_THRESHOLDS = {
    "high":   0.65,
    "medium": 0.40,
}


def _label_overlap(a: str, b: str) -> bool:
    """Return True if two condition labels likely refer to the same condition."""
    stop = {"disease", "condition", "infection", "the", "of"}
    return bool(set(a.split("_")) & set(b.split("_")) - stop)


def _determine_urgency(
    sym_pred: str,
    sym_conf: float,
    agreement: str,
    uncertainty: str,
    has_image: bool,
    img_pred: str,
    img_conf: float,
) -> str:
    for pred in (sym_pred, img_pred if has_image else ""):
        if pred in _URGENT_CONDITIONS:
            return "urgent"
    for pred in (sym_pred, img_pred if has_image else ""):
        if pred in _VET_SOON_CONDITIONS:
            return "vet_soon"
    if agreement == "conflict" or uncertainty == "out_of_scope":
        return "vet_soon"
    if sym_conf >= _CONFIDENCE_THRESHOLDS["high"] and uncertainty == "confident":
        return "monitor"
    if sym_conf >= _CONFIDENCE_THRESHOLDS["medium"]:
        return "monitor"
    return "monitor"


def _build_recommendation(urgency_level: str, uncertainty_status: str,
                           prediction: str, agreement: str) -> str:
    display = prediction.replace("_", " ").title() if prediction not in ("unknown", "") else "an unknown condition"
    if urgency_level == "urgent":
        return (f"This case shows signs consistent with {display}. "
                "Please seek emergency veterinary care immediately.")
    if urgency_level == "vet_soon":
        if agreement == "conflict":
            return ("The symptom and image assessments give conflicting signals. "
                    "Please have your pet examined by a veterinarian as soon as possible.")
        return (f"Signs are consistent with {display}. "
                "Please schedule a veterinary appointment within 24–48 hours.")
    if urgency_level == "monitor":
        if uncertainty_status in ("uncertain", "conflict", "out_of_scope"):
            return (f"Possible {display} detected, but confidence is low. "
                    "Monitor your pet closely and consult a veterinarian if symptoms worsen.")
        return (f"Signs may be consistent with {display}. "
                "Monitor your pet and consult a veterinarian if symptoms persist or worsen.")
    return ("No immediate concern detected. "
            "Continue monitoring your pet and consult a veterinarian for routine care.")


def _build_reasoning(sym_pred: str, sym_conf: float, img_pred: str,
                     img_conf: float, agreement: str, uncertainty: str,
                     has_image: bool) -> str:
    sym_part = f"Symptom model → {sym_pred} ({sym_conf:.0%})"
    if has_image and img_pred:
        return (f"{sym_part}; Image model → {img_pred} ({img_conf:.0%}); "
                f"Agreement={agreement}; Uncertainty={uncertainty}.")
    return f"{sym_part}; No image; Uncertainty={uncertainty}."


def _classify_urgency(symptom_result: Dict[str, Any],
                      image_result:   Dict[str, Any]) -> Dict[str, Any]:
    """
    Pure-Python deterministic urgency classifier.

    Args:
        symptom_result: Output dict from Agent 2.
        image_result:   Output dict from Agent 3 (may be empty).

    Returns:
        Dict with urgency_level, recommendation, uncertainty_status,
        agreement_status, and reasoning.
    """
    sym_pred:   str   = (symptom_result.get("top_prediction") or "unknown").lower().replace(" ", "_")
    sym_conf:   float = float(symptom_result.get("confidence") or 0.0)
    sym_unc:    bool  = bool(symptom_result.get("uncertainty_flag", False))
    sym_oos:    bool  = bool(symptom_result.get("possible_out_of_scope", False))
    sym_status: str   = symptom_result.get("assessment_status", "completed")

    img_pred:   str   = (image_result.get("image_prediction") or "").lower().replace(" ", "_")
    img_conf:   float = float(image_result.get("confidence") or 0.0)
    img_unc:    bool  = bool(image_result.get("uncertainty_flag", False))
    img_valid:  str   = image_result.get("image_validity", "none")
    has_image:  bool  = bool(image_result) and img_valid not in ("none", "unusable", "error", "")

    # Agreement
    if not has_image or sym_status != "completed":
        agreement = "symptom_only"
    elif sym_pred and img_pred:
        agreement = "agree" if _label_overlap(sym_pred, img_pred) or sym_pred == img_pred else "conflict"
    else:
        agreement = "none"

    # Uncertainty
    if sym_oos:
        uncertainty = "out_of_scope"
    elif agreement == "conflict":
        uncertainty = "conflict"
    elif sym_unc or (has_image and img_unc):
        uncertainty = "uncertain"
    else:
        uncertainty = "confident"

    urgency = _determine_urgency(sym_pred, sym_conf, agreement, uncertainty,
                                  has_image, img_pred, img_conf)
    recommendation = _build_recommendation(urgency, uncertainty, sym_pred, agreement)
    reasoning = _build_reasoning(sym_pred, sym_conf, img_pred, img_conf,
                                  agreement, uncertainty, has_image)

    return {
        "urgency_level":      urgency,
        "recommendation":     recommendation,
        "uncertainty_status": uncertainty,
        "agreement_status":   agreement,
        "reasoning":          reasoning,
    }


# ---------------------------------------------------------------------------
# CrewAI tool wrapper
# ---------------------------------------------------------------------------

class UrgencyCalculatorTool(BaseTool):
    """
    Deterministic urgency calculator for the Pet Health Triage Agent.

    Accepts a JSON string with keys:
        - symptom_result (dict): output dict from Agent 2
        - image_result   (dict): output dict from Agent 3 (may be empty or absent)

    Returns a JSON string with:
        - urgency_level      : "non-urgent" | "monitor" | "vet_soon" | "urgent"
        - recommendation     : safe plain-text advice
        - uncertainty_status : "confident" | "uncertain" | "conflict" | "out_of_scope"
        - agreement_status   : "agree" | "conflict" | "symptom_only" | "none"
        - reasoning          : brief one-sentence explanation
    """

    name: str = "urgency_calculator"
    description: str = (
        "Calculates an urgency level and recommendation by combining symptom "
        "and image assessment results. Input must be a JSON string with keys "
        "'symptom_result' (dict) and optionally 'image_result' (dict)."
    )

    def _run(self, input_str: str) -> str:
        """
        Execute urgency calculation.

        Args:
            input_str: JSON string with 'symptom_result' and optional 'image_result'.

        Returns:
            JSON string with urgency_level, recommendation, uncertainty_status,
            agreement_status, and reasoning.
        """
        try:
            payload: Dict[str, Any] = json.loads(input_str) if isinstance(input_str, str) else input_str
        except json.JSONDecodeError as exc:
            logger.warning("UrgencyCalculatorTool: invalid JSON input: %s", exc)
            payload = {}

        symptom_result: Dict[str, Any] = payload.get("symptom_result") or {}
        image_result:   Dict[str, Any] = payload.get("image_result")   or {}

        try:
            result = _classify_urgency(symptom_result, image_result)
        except Exception as exc:
            logger.error("UrgencyCalculatorTool failed: %s", exc, exc_info=True)
            result = {
                "urgency_level":      "monitor",
                "recommendation":     "Unable to compute urgency. Please consult a veterinarian.",
                "uncertainty_status": "uncertain",
                "agreement_status":   "none",
                "reasoning":          f"Tool error: {exc}",
            }

        return json.dumps(result)