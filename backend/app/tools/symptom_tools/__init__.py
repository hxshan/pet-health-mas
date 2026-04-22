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
    # individual kwargs are passed by the LLM automatically

Option B — direct Python call (standalone / testing):
    from app.tools.symptom_tools import predict, analyse
    result   = predict(raw_dict)
    analysis = analyse(result)
"""
import json
from typing import Optional

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from app.tools.symptom_tools.predictor  import predict, PredictionResult  # noqa: F401
from app.tools.symptom_tools.confidence import analyse                     # noqa: F401


# ---------------------------------------------------------------------------
# Input schema — ALL fields optional so partial LLM calls are tolerated.
# The predictor already fills missing feature columns with 0.
# ---------------------------------------------------------------------------
class SymptomInput(BaseModel):
    # --- pet profile (strings) ---
    species:    Optional[str]   = Field(default="unknown", description="Pet species, e.g. 'dog' or 'cat'")
    breed:      Optional[str]   = Field(default="unknown", description="Pet breed, e.g. 'labrador'")
    sex:        Optional[str]   = Field(default="unknown", description="'male' or 'female'")
    neutered:   Optional[str]   = Field(default="unknown", description="'yes' or 'no'")
    age_years:  Optional[float] = Field(default=0.0,       description="Age of pet in years")
    weight_kg:  Optional[float] = Field(default=0.0,       description="Weight of pet in kilograms")

    # --- visit history (int) ---
    vaccinated:            Optional[int] = Field(default=0,  description="1 if vaccinated, else 0")
    num_previous_visits:   Optional[int] = Field(default=0,  description="Number of previous vet visits")
    prev_diagnosis_class:  Optional[int] = Field(default=-1, description="Previous diagnosis class index")
    days_since_last_visit: Optional[int] = Field(default=0,  description="Days since last vet visit")
    chronic_flag:          Optional[int] = Field(default=0,  description="1 if chronic condition, else 0")

    # --- gastro / general (int) ---
    vomiting:      Optional[int] = Field(default=0, description="1 if present, else 0")
    diarrhea:      Optional[int] = Field(default=0, description="1 if present, else 0")
    fever:         Optional[int] = Field(default=0, description="1 if present, else 0")
    lethargy:      Optional[int] = Field(default=0, description="1 if present, else 0")
    loss_appetite: Optional[int] = Field(default=0, description="1 if present, else 0")
    dehydration:   Optional[int] = Field(default=0, description="1 if present, else 0")

    # --- skin (int) ---
    itching:      Optional[int] = Field(default=0, description="1 if present, else 0")
    red_skin:     Optional[int] = Field(default=0, description="1 if present, else 0")
    hair_loss:    Optional[int] = Field(default=0, description="1 if present, else 0")
    skin_lesions: Optional[int] = Field(default=0, description="1 if present, else 0")
    wounds:       Optional[int] = Field(default=0, description="1 if present, else 0")

    # --- tick fever (int) ---
    dark_urine:    Optional[int] = Field(default=0, description="1 if present, else 0")
    pale_gums:     Optional[int] = Field(default=0, description="1 if present, else 0")
    pale_eyelids:  Optional[int] = Field(default=0, description="1 if present, else 0")
    tick_exposure: Optional[int] = Field(default=0, description="1 if present, else 0")

    # --- UTI (int) ---
    pain_urinating:     Optional[int] = Field(default=0, description="1 if present, else 0")
    frequent_urination: Optional[int] = Field(default=0, description="1 if present, else 0")
    blood_in_urine:     Optional[int] = Field(default=0, description="1 if present, else 0")


_INT_FLAGS = {
    "vaccinated", "vomiting", "diarrhea", "fever", "lethargy", "loss_appetite",
    "dehydration", "itching", "red_skin", "hair_loss", "skin_lesions", "wounds",
    "dark_urine", "pale_gums", "pale_eyelids", "tick_exposure",
    "pain_urinating", "frequent_urination", "blood_in_urine",
}


class SymptomClassifierTool(BaseTool):
    """
    CrewAI tool — classifies a pet's symptoms via a local XGBoost ML model.

    Pass each field as a separate argument.  All fields have defaults so you
    only need to supply the ones that are known.  Symptom flags accept 0/1 as
    integer or string.

    Returns JSON with:
        top_prediction, confidence, top3, alternatives,
        uncertainty_flag, uncertainty_reason, top_gap, possible_out_of_scope
    """

    name: str = "symptom_classifier"
    description: str = (
        "Classify a pet's symptoms using a local XGBoost ML model. "
        "Pass known profile fields (species, breed, sex, neutered, age_years, weight_kg) "
        "and any symptom flags that are present as 1 (omit or use 0 for absent). "
        "All fields are optional — supply as many as you know. "
        "Returns top_prediction, confidence, uncertainty_flag."
    )
    args_schema: type[BaseModel] = SymptomInput

    def _run(self, **kwargs) -> str:
        # Unwrap if CrewAI passes the whole case as {"object": {...}}
        if "object" in kwargs and isinstance(kwargs["object"], dict):
            kwargs = kwargs["object"]

        # Coerce string "0"/"1" flags → int
        raw = {}
        for k, v in kwargs.items():
            if k in _INT_FLAGS:
                try:
                    raw[k] = int(v)
                except (TypeError, ValueError):
                    raw[k] = 0
            else:
                raw[k] = v

        # ── Missing-data guard ─────────────────────────────────────────────────
        missing = []
        for field in ("species", "sex", "neutered"):
            if not raw.get(field) or raw.get(field) == "unknown":
                missing.append(field)
        for field in ("age_years", "weight_kg"):
            val = raw.get(field)
            if val is None or float(val) == 0.0:
                missing.append(field)

        if missing:
            return json.dumps({
                "assessment_status": "needs_more_info",
                "needs_more_info":   True,
                "missing_fields":    missing,
                "uncertainty_flag":  True,
                "uncertainty_reason": "insufficient input data",
            })
        # ── End missing-data guard ─────────────────────────────────────────────

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
