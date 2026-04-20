"""
Symptom Classifier Tool — stub
================================
TODO (Symptom Assessment team):
  - Replace `_run` with a real trained ML model (sklearn, pytorch, etc.)
  - Load the model artifact from a local path (e.g. models/symptom_model.pkl)
  - Keep the output schema exactly as documented below so other agents work.

Output schema from _run():
{
  "top_prediction": str,        # e.g. "skin_infection"
  "confidence": float,          # 0.0 – 1.0
  "alternatives": [             # top-k other predictions
      {"condition": str, "confidence": float}
  ],
  "uncertainty_flag": bool,     # true when confidence is low or gap is small
  "possible_out_of_scope": bool
}
"""
import json
from typing import Any, Dict, List

from crewai.tools import BaseTool


class SymptomClassifierTool(BaseTool):
    name: str = "symptom_classifier"
    description: str = (
        "Classify a list of pet symptoms using a local ML model. "
        "Input JSON: {\"symptoms\": [\"vomiting\", \"lethargy\"]}. "
        "Returns top_prediction, confidence, alternatives, uncertainty_flag."
    )

    def _run(self, input_str: str) -> str:
        # TODO: replace with real model inference
        try:
            symptoms = json.loads(input_str).get("symptoms", [])
        except Exception:
            symptoms = []

        # Stub result — always uncertain so triage stays safe
        result: Dict[str, Any] = {
            "top_prediction": "unknown",
            "confidence": 0.0,
            "alternatives": [],
            "uncertainty_flag": True,
            "possible_out_of_scope": True,
        }
        return json.dumps(result)
