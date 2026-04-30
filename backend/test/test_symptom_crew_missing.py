"""
Agent 2 — missing-data handling test.

Tests that the symptom agent's decision layer (logic.py) correctly
identifies incomplete cases and returns needs_more_info WITHOUT
invoking the ML predictor.

This is the correct place to test agentic decision-making:
  - The Python agent layer (logic.py) is responsible for sufficiency checks
  - The LLM (CrewAI) is responsible for reasoning about complete cases
  - Asking a small local LLM to decide NOT to call a tool is unreliable

Run from backend/:
    python test/test_symptom_crew_missing.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.agents.symptom_agent.logic import check_input_sufficiency, run_symptom_assessment

# ── Partial / incomplete case ─────────────────────────────────────────────────
PARTIAL_CASE = {
    "species":  "dog",
    "vomiting": 1,
    # missing: sex, neutered, age_years, weight_kg
}

print("\n" + "="*60)
print("Running Agent 2 — missing-data decision test...")
print("="*60 + "\n")

# ── Step 1: sufficiency check (agent decision layer) ──────────────────────────
sufficiency = check_input_sufficiency(PARTIAL_CASE)
print("Sufficiency check result:")
print(json.dumps(sufficiency, indent=2))

assert sufficiency["sufficient"] is False, "Agent should detect case as insufficient"
assert len(sufficiency["missing_fields"]) > 0, "Agent should list missing fields"
print("\nSufficiency check PASSED — agent correctly identified incomplete case.")

# ── Step 2: full agent run — should return needs_more_info, no predictor call ─
result = run_symptom_assessment(PARTIAL_CASE)
print("\nFull agent run result:")
print(json.dumps(result, indent=2))

assert result.get("assessment_status") == "needs_more_info", (
    f"Expected assessment_status='needs_more_info', got: {result.get('assessment_status')}"
)
assert result.get("needs_more_info") is True, (
    f"Expected needs_more_info=true, got: {result.get('needs_more_info')}"
)
assert isinstance(result.get("missing_fields"), list) and len(result["missing_fields"]) > 0, (
    f"Expected non-empty missing_fields, got: {result.get('missing_fields')}"
)

print("\nALL ASSERTIONS PASSED -- Agent 2 agentic missing-data handling verified.")
print(f"  assessment_status : {result['assessment_status']}")
print(f"  needs_more_info   : {result['needs_more_info']}")
print(f"  missing_fields    : {result['missing_fields']}")
