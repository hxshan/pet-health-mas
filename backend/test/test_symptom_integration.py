"""
Agent 2 — CrewAI + LLM + Tool Integration Tests
=================================================
Tests the real end-to-end pipeline:
  CrewAI → llama3.1:8b (Ollama) → symptom_classifier tool → JSON output

These tests are SLOW (~30–90s each) because they call the live LLM.
They are NOT deterministic — LLM output may vary between runs.

Assertions are intentionally lenient: we validate structure, not exact values.

Run from backend/:
    python test/test_symptom_integration.py
"""
import sys
import os
import json
import re

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crewai import Crew, Task, Process
from app.crew.agents import make_symptom_agent


# ── Shared task builder ────────────────────────────────────────────────────────

def make_task_complete(agent, case: dict) -> Task:
    """Task for a complete case — just call the tool and return its output."""
    return Task(
        description=(
            "Call the symptom_classifier tool with the case data below "
            "and return its JSON output as your Final Answer.\n\n"
            f"Case: {json.dumps(case)}"
        ),
        expected_output=(
            "The exact JSON returned by the symptom_classifier tool. "
            "Must contain: status, top_prediction, confidence, uncertainty_flag."
        ),
        agent=agent,
    )


def make_task_incomplete(agent, case: dict) -> Task:
    """Task for an incomplete case — tool will block and return needs_more_info."""
    return Task(
        description=(
            "Call the symptom_classifier tool with the case data below "
            "and return its JSON output as your Final Answer.\n\n"
            f"Case: {json.dumps(case)}"
        ),
        expected_output=(
            "The exact JSON returned by the symptom_classifier tool. "
            "Since required fields are missing, it will contain: "
            "assessment_status, needs_more_info, missing_fields."
        ),
        agent=agent,
    )


def extract_json(text: str) -> dict:
    """
    Try to parse JSON from LLM output.
    Handles cases where the LLM wraps JSON in markdown code blocks or adds prose.
    """
    # Direct parse
    try:
        return json.loads(str(text))
    except json.JSONDecodeError:
        pass

    # Extract first {...} block
    match = re.search(r'\{.*\}', str(text), re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {}


def run_crew(case: dict, complete: bool = True) -> tuple[dict, str]:
    """Spin up a fresh agent+crew for one test run. Returns (parsed_output, raw_str)."""
    agent = make_symptom_agent()
    task  = make_task_complete(agent, case) if complete else make_task_incomplete(agent, case)
    crew  = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=True)
    result = crew.kickoff()
    raw    = str(result)
    return extract_json(raw), raw


# ══════════════════════════════════════════════════════════════════════════════
# Test 1 — Happy path: complete case → tool should fire → prediction JSON
# ══════════════════════════════════════════════════════════════════════════════

COMPLETE_CASE = {
    "species":  "dog",
    "breed":    "labrador",
    "sex":      "male",
    "neutered": "yes",
    "age_years": 3.0,
    "weight_kg": 25.0,
    "vaccinated":            1,
    "num_previous_visits":   0,
    "prev_diagnosis_class":  -1,
    "days_since_last_visit": 0,
    "chronic_flag":          0,
    "vomiting":      1,
    "diarrhea":      1,
    "fever":         0,
    "lethargy":      1,
    "loss_appetite": 1,
    "dehydration":   0,
    "itching":       0,
    "red_skin":      0,
    "hair_loss":     0,
    "skin_lesions":  0,
    "wounds":        0,
    "dark_urine":    0,
    "pale_gums":     0,
    "pale_eyelids":  0,
    "tick_exposure": 0,
    "pain_urinating":     0,
    "frequent_urination": 0,
    "blood_in_urine":     0,
}

print("\n" + "="*60)
print("INTEGRATION TEST 1 — Happy path (complete case)")
print("="*60)

output1, raw1 = run_crew(COMPLETE_CASE, complete=True)

print("\n--- RAW LLM OUTPUT ---")
print(raw1)
print("\n--- PARSED OUTPUT ---")
print(json.dumps(output1, indent=2))

# Lenient assertions: tool must have fired and returned prediction fields
assert "top_prediction" in output1, (
    f"Expected 'top_prediction' in output. Got keys: {list(output1.keys())}\nRaw: {raw1}"
)
assert "confidence" in output1, (
    f"Expected 'confidence' in output. Got keys: {list(output1.keys())}\nRaw: {raw1}"
)
assert "uncertainty_flag" in output1, (
    f"Expected 'uncertainty_flag' in output. Got keys: {list(output1.keys())}\nRaw: {raw1}"
)
assert isinstance(output1["confidence"], (int, float)), (
    f"Expected confidence to be a number, got: {type(output1['confidence'])}"
)

print("\nTEST 1 PASSED -- tool fired, prediction returned.")
print(f"  top_prediction  : {output1.get('top_prediction')}")
print(f"  confidence      : {output1.get('confidence')}")
print(f"  uncertainty_flag: {output1.get('uncertainty_flag')}")


# ══════════════════════════════════════════════════════════════════════════════
# Test 2 — Missing-data path: incomplete case → tool blocked → needs_more_info
# Note: the tool hard-blocks incomplete cases even if the LLM calls it,
#       so this test verifies the system-level behaviour regardless of
#       whether the LLM or the tool made the decision.
# ══════════════════════════════════════════════════════════════════════════════

INCOMPLETE_CASE = {
    "species":  "dog",
    "vomiting": 1,
    # missing: sex, neutered, age_years, weight_kg
}

print("\n" + "="*60)
print("INTEGRATION TEST 2 — Missing-data path (incomplete case)")
print("="*60)

output2, raw2 = run_crew(INCOMPLETE_CASE, complete=False)

print("\n--- RAW LLM OUTPUT ---")
print(raw2)
print("\n--- PARSED OUTPUT ---")
print(json.dumps(output2, indent=2))

# Lenient assertions: system must surface needs_more_info regardless of path
assert output2.get("assessment_status") == "needs_more_info", (
    f"Expected assessment_status='needs_more_info'. Got: {output2.get('assessment_status')}\nRaw: {raw2}"
)
assert output2.get("needs_more_info") is True, (
    f"Expected needs_more_info=true. Got: {output2.get('needs_more_info')}\nRaw: {raw2}"
)
assert isinstance(output2.get("missing_fields"), list) and len(output2["missing_fields"]) > 0, (
    f"Expected non-empty missing_fields. Got: {output2.get('missing_fields')}\nRaw: {raw2}"
)

print("\nTEST 2 PASSED -- needs_more_info returned for incomplete case.")
print(f"  assessment_status : {output2.get('assessment_status')}")
print(f"  needs_more_info   : {output2.get('needs_more_info')}")
print(f"  missing_fields    : {output2.get('missing_fields')}")


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ALL INTEGRATION TESTS PASSED")
print("="*60)
