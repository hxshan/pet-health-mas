"""
symptom_tools
=============
Public API for the Symptom Assessment Agent's ML tooling.

Package layout
--------------
  predictor.py   — XGBoost model load + raw inference  (predict())
  confidence.py  — uncertainty / ambiguity analysis     (analyse())
  __init__.py    — CrewAI BaseTool wrapper + re-exports

Agent 2 usage
-------------
Option A — via CrewAI (inside the Crew):
    tool = SymptomClassifierTool()
    result_json = tool.run(json.dumps({...}))

Option B — direct Python call (standalone / testing):
    from app.tools.symptom_tools import predict, analyse
    result   = predict(raw_dict)
    analysis = analyse(result)
"""
import json
from typing import Any, Dict

from crewai.tools import BaseTool

from app.tools.symptom_tools.predictor  import predict, PredictionResult  # noqa: F401
from app.tools.symptom_tools.confidence import analyse                     # noqa: F401


class SymptomClassifierTool(BaseTool):
    """
    CrewAI tool wrapper around the local XGBoost symptom classifier.

    Input (JSON string) — required fields:
        species, breed, sex, neutered, age_years, weight_kg

    Optional binary symptom flags (0 or 1):
        vomiting, diarrhea, fever, lethargy, loss_appetite, dehydration,
        itching, red_skin, hair_loss, skin_lesions, wounds,
        dark_urine, pale_gums, pale_eyelids, tick_exposure,
        pain_urinating, frequent_urination, blood_in_urine

    Output (JSON string):
        {
            "status":             "predicted" | "uncertain" | "error",
            "top_prediction":     str,
            "confidence":         float,
            "top3":               [{"condition": str, "confidence": float}],
            "alternatives":       [{"condition": str, "confidence": float}],
            "probability_map":    {condition: float},
            "uncertainty_flag":   bool,
            "uncertainty_reason": str,
            "top_gap":            float,
            "possible_out_of_scope": bool
        }
    """

    name: str = "symptom_classifier"
    description: str = (
        "Classify a pet's symptoms using a local XGBoost ML model. "
        "Input: JSON string with pet profile and binary symptom flags. "
        "Required: species, breed, sex, neutered, age_years, weight_kg. "
        "Returns top_prediction, confidence, alternatives, uncertainty_flag."
    )

    def _run(self, input_str: str) -> str:
        try:
            raw: Dict[str, Any] = json.loads(input_str)
        except json.JSONDecodeError as e:
            return json.dumps({"status": "error", "error": f"Invalid JSON: {e}"})

        try:
            result   = predict(raw)
            analysis = analyse(result)
        except RuntimeError as e:
            return json.dumps({"status": "error", "error": str(e)})

        output = {
            "status":          "uncertain" if analysis["uncertainty_flag"] else "predicted",
            "top_prediction":  result.top_label,
            "confidence":      result.top_confidence,
            "top3":            result.top3,
            "alternatives":    result.top3[1:],
            "probability_map": result.probability_map,
            **analysis,
        }
        return json.dumps(output)
