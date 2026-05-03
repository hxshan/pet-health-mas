"""
Agent 2 — Symptom Assessment Unit Tests
=========================================
Tests the core logic layer directly (no LLM, no CrewAI).

Run from backend/:
    python -m pytest tests/test_symptom_agent.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.agents.symptom_agent.logic import run_symptom_assessment, check_input_sufficiency

# ── Shared fixtures ────────────────────────────────────────────────────────────

COMPLETE_CASE = {
    "species":   "dog",
    "breed":     "labrador",
    "sex":       "male",
    "neutered":  "yes",
    "age_years": 3.0,
    "weight_kg": 25.0,
    "vaccinated": 1, "num_previous_visits": 0,
    "prev_diagnosis_class": -1, "days_since_last_visit": 0, "chronic_flag": 0,
    # gastro symptoms
    "vomiting": 1, "diarrhea": 1, "lethargy": 1, "loss_appetite": 1,
    "fever": 0, "dehydration": 0,
    "itching": 0, "red_skin": 0, "hair_loss": 0, "skin_lesions": 0, "wounds": 0,
    "dark_urine": 0, "pale_gums": 0, "pale_eyelids": 0, "tick_exposure": 0,
    "pain_urinating": 0, "frequent_urination": 0, "blood_in_urine": 0,
}

INCOMPLETE_CASE = {
    "species":  "dog",
    "vomiting": 1,
    # missing: age_years, weight_kg
}


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestSymptomAgent:

    def test_complete_case_returns_prediction(self):
        """Happy path — full case produces a completed assessment with a prediction."""
        result = run_symptom_assessment(COMPLETE_CASE)

        assert result["assessment_status"] == "completed"
        assert result["top_prediction"] not in (None, "", "unknown")
        assert isinstance(result["confidence"], float)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_missing_required_fields_returns_needs_more_info(self):
        """Incomplete case must be blocked before the model runs."""
        result = run_symptom_assessment(INCOMPLETE_CASE)

        assert result["assessment_status"] == "needs_more_info"
        assert result["needs_more_info"] is True
        assert isinstance(result["missing_fields"], list)
        assert len(result["missing_fields"]) > 0

    def test_no_symptoms_triggers_needs_more_info(self):
        """All symptom flags = 0 counts as insufficient even if profile is complete."""
        no_symptoms = {**COMPLETE_CASE, **{
            "vomiting": 0, "diarrhea": 0, "lethargy": 0, "loss_appetite": 0,
        }}
        result = run_symptom_assessment(no_symptoms)

        assert result["assessment_status"] == "needs_more_info"
        assert "at_least_one_symptom_flag" in result.get("missing_fields", [])

    def test_output_contains_required_keys(self):
        """Response schema must always contain the full set of expected keys."""
        result = run_symptom_assessment(COMPLETE_CASE)

        required_keys = {
            "assessment_status", "top_prediction", "confidence",
            "uncertainty_flag", "alternatives", "possible_out_of_scope",
            "needs_more_info", "missing_fields",
        }
        assert required_keys.issubset(result.keys())

    def test_sufficiency_check_lists_missing_fields(self):
        """check_input_sufficiency correctly identifies which fields are absent."""
        result = check_input_sufficiency(INCOMPLETE_CASE)

        assert result["sufficient"] is False
        missing = result["missing_fields"]
        assert "age_years" in missing
        assert "weight_kg" in missing
