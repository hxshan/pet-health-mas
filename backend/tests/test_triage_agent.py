"""
Evaluation Script — Triage & Case Synthesis Agent (Agent 4)
============================================================
Tests UrgencyCalculatorTool and triage_agent.run().

Coverage:
  1.  Schema validation      — triage_result always has all required keys
  2.  Final report keys      — final_report always has all required keys
  3.  Urgent condition       — known urgent condition → urgency_level="urgent"
  4.  Vet-soon condition     — known vet_soon condition → urgency_level="vet_soon"
  5.  High confidence        — high conf + no uncertainty → urgency="monitor"
  6.  Conflict detection     — symptom ≠ image prediction → agreement="conflict"
  7.  Agreement detection    — symptom == image prediction → agreement="agree"
  8.  No image               — empty image_result → agreement="symptom_only"
  9.  Unusable image         — image_validity="unusable" → agreement="symptom_only"
  10. Out-of-scope flag      — possible_out_of_scope=True → uncertainty="out_of_scope"
  11. Uncertainty propagated — sym uncertainty_flag=True → uncertainty_status≠confident
  12. State keys written     — agent run() writes triage_result + final_report to state
  13. Disclaimer present     — final_report always contains disclaimer
  14. Tool invalid JSON      — malformed input → safe fallback, no exception
  15. Observability trace    — trace_events appended after run()

Run with:
    cd backend
    python -m pytest tests/test_triage_agent.py -v
"""
from __future__ import annotations

import json
import unittest
from typing import Any, Dict
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Required schema keys
# ---------------------------------------------------------------------------

TRIAGE_RESULT_KEYS = {
    "primary_assessment",
    "urgency_level",
    "recommendation",
    "uncertainty_status",
    "agreement_status",
}

FINAL_REPORT_KEYS = {
    "triage_summary",
    "recommendations",
    "disclaimer",
}

VALID_URGENCY_LEVELS = {"non-urgent", "monitor", "vet_soon", "urgent"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_symptom(prediction: str = "skin_infection", confidence: float = 0.75,
                   uncertainty: bool = False, oos: bool = False,
                   status: str = "completed") -> Dict[str, Any]:
    return {
        "top_prediction":       prediction,
        "confidence":           confidence,
        "alternatives":         [],
        "uncertainty_flag":     uncertainty,
        "possible_out_of_scope": oos,
        "assessment_status":    status,
    }


def _make_image(prediction: str = "skin_infection", confidence: float = 0.80,
                validity: str = "valid", uncertainty: bool = False) -> Dict[str, Any]:
    return {
        "image_prediction": prediction,
        "confidence":       confidence,
        "image_validity":   validity,
        "uncertainty_flag": uncertainty,
        "alternatives":     [],
    }


def _run_tool(symptom: Dict, image: Dict = None) -> Dict[str, Any]:
    from app.tools.triage_tools import UrgencyCalculatorTool
    tool = UrgencyCalculatorTool()
    payload = json.dumps({"symptom_result": symptom, "image_result": image or {}})
    return json.loads(tool._run(payload))


def _run_agent(symptom: Dict, image: Dict = None,
               pet_profile: Dict = None) -> Dict[str, Any]:
    from app.agents.triage_agent.agent import run
    state: Dict[str, Any] = {
        "symptom_assessment": symptom,
        "image_assessment":   image or {},
        "pet_profile":        pet_profile or {"species": "dog"},
        "trace_events":       [],
    }
    return run(state)


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

class TestTriageToolSchema(unittest.TestCase):

    def test_01_triage_result_has_required_keys(self):
        """triage_result must always contain all required keys."""
        result = _run_tool(_make_symptom())
        for key in ("urgency_level", "recommendation", "uncertainty_status",
                    "agreement_status", "reasoning"):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_02_final_report_has_required_keys(self):
        """agent run() must produce final_report with required keys."""
        state = _run_agent(_make_symptom())
        for key in FINAL_REPORT_KEYS:
            self.assertIn(key, state["final_report"], f"Missing final_report key: {key}")

    def test_03_urgency_level_is_valid(self):
        """urgency_level must be one of the four valid values."""
        result = _run_tool(_make_symptom())
        self.assertIn(result["urgency_level"], VALID_URGENCY_LEVELS)

    def test_04_urgent_condition_maps_to_urgent(self):
        """Known urgent condition should produce urgency_level='urgent'."""
        result = _run_tool(_make_symptom(prediction="poisoning", confidence=0.9))
        self.assertEqual(result["urgency_level"], "urgent")

    def test_05_vet_soon_condition(self):
        """Known vet_soon condition should produce urgency_level='vet_soon'."""
        result = _run_tool(_make_symptom(prediction="parvovirus", confidence=0.8))
        self.assertEqual(result["urgency_level"], "vet_soon")

    def test_06_high_confidence_gives_monitor(self):
        """High confidence non-critical condition → monitor."""
        result = _run_tool(_make_symptom(prediction="mild_dermatitis", confidence=0.80))
        self.assertEqual(result["urgency_level"], "monitor")

    def test_07_conflict_detection(self):
        """Symptom and image predict different conditions → agreement='conflict'."""
        sym = _make_symptom(prediction="skin_infection",   confidence=0.70)
        img = _make_image(prediction="ear_infection",      confidence=0.70)
        result = _run_tool(sym, img)
        self.assertEqual(result["agreement_status"], "conflict")

    def test_08_agreement_detection(self):
        """Same prediction from symptom and image → agreement='agree'."""
        sym = _make_symptom(prediction="skin_infection", confidence=0.70)
        img = _make_image(prediction="skin_infection",   confidence=0.75)
        result = _run_tool(sym, img)
        self.assertEqual(result["agreement_status"], "agree")

    def test_09_no_image_gives_symptom_only(self):
        """Empty image_result → agreement_status='symptom_only'."""
        result = _run_tool(_make_symptom(), {})
        self.assertEqual(result["agreement_status"], "symptom_only")

    def test_10_unusable_image_gives_symptom_only(self):
        """image_validity='unusable' → treated as no image."""
        img = _make_image(validity="unusable")
        result = _run_tool(_make_symptom(), img)
        self.assertEqual(result["agreement_status"], "symptom_only")

    def test_11_out_of_scope_flag(self):
        """possible_out_of_scope=True → uncertainty_status='out_of_scope'."""
        sym = _make_symptom(oos=True)
        result = _run_tool(sym)
        self.assertEqual(result["uncertainty_status"], "out_of_scope")

    def test_12_uncertainty_flag_propagated(self):
        """symptom uncertainty_flag=True → uncertainty_status != 'confident'."""
        sym = _make_symptom(uncertainty=True, confidence=0.35)
        result = _run_tool(sym)
        self.assertNotEqual(result["uncertainty_status"], "confident")

    def test_13_state_keys_written(self):
        """agent run() must write triage_result and final_report to state."""
        state = _run_agent(_make_symptom())
        self.assertIn("triage_result", state)
        self.assertIn("final_report", state)
        for key in TRIAGE_RESULT_KEYS:
            self.assertIn(key, state["triage_result"], f"Missing triage_result key: {key}")

    def test_14_disclaimer_present(self):
        """final_report must always include a disclaimer string."""
        state = _run_agent(_make_symptom())
        disclaimer = state["final_report"].get("disclaimer", "")
        self.assertIsInstance(disclaimer, str)
        self.assertGreater(len(disclaimer), 10)

    def test_15_invalid_json_safe_fallback(self):
        """Malformed JSON input must not raise — must return safe fallback."""
        from app.tools.triage_tools import UrgencyCalculatorTool
        tool = UrgencyCalculatorTool()
        result_raw = tool._run("this is not json {{{{")
        result = json.loads(result_raw)
        self.assertIn(result["urgency_level"], VALID_URGENCY_LEVELS)
        self.assertIn("recommendation", result)

    def test_16_observability_trace_appended(self):
        """trace_events must have at least one triage event after run()."""
        state = _run_agent(_make_symptom())
        events = state.get("trace_events", [])
        self.assertTrue(len(events) >= 1)
        event_types = [e.get("event_type") for e in events]
        self.assertIn("triage_agent_completed", event_types)


if __name__ == "__main__":
    unittest.main()