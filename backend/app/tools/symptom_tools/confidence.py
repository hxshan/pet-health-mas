"""
Confidence and uncertainty analysis for symptom predictions.

Agent 2 can call these functions directly after getting a PredictionResult
from predictor.predict() — no need to go through the CrewAI tool.

Thresholds are intentionally conservative so uncertain cases are always
escalated to a vet rather than silently passed.
"""
from typing import Any, Dict

from app.tools.symptom_tools.predictor import PredictionResult

# Below this → low confidence
CONFIDENCE_THRESHOLD: float = 0.35

# Top-2 gap below this → ambiguous (too close to call)
AMBIGUITY_GAP_THRESHOLD: float = 0.10


def analyse(result: PredictionResult) -> Dict[str, Any]:
    """
    Evaluate a PredictionResult and return an uncertainty analysis dict.

    This is intentionally separate from the model so Agent 2 can:
      - call predict() to get raw output
      - call analyse() to reason about reliability
      - then apply its own LLM-based interpretation on top

    Returns:
        {
            "uncertainty_flag":      bool,
            "possible_out_of_scope": bool,   # always False here — Agent 2 decides this
            "uncertainty_reason":    str,    # human-readable explanation
            "top_gap":               float,  # confidence gap between top-1 and top-2
        }
    """
    top_conf = result.top_confidence
    top_gap  = (
        top_conf - result.top3[1]["confidence"]
        if len(result.top3) > 1
        else 1.0
    )

    uncertain = False
    reasons   = []

    if top_conf < CONFIDENCE_THRESHOLD:
        uncertain = True
        reasons.append(f"low confidence result ({top_conf:.2%})")

    if top_gap < AMBIGUITY_GAP_THRESHOLD:
        uncertain = True
        reasons.append(
            f"top-2 predictions are too close "
            f"({result.top3[0]['condition']} vs {result.top3[1]['condition']}, "
            f"gap={top_gap:.2%})"
        )

    return {
        "uncertainty_flag":      uncertain,
        "possible_out_of_scope": False,   # Agent 2 sets this based on intake data
        "uncertainty_reason":    "; ".join(reasons) if reasons else "within supported conditions",
        "top_gap":               round(top_gap, 4),
    }
