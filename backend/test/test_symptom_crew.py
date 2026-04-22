"""
CrewAI integration test for Agent 2 (Symptom Assessment Agent).

Runs Agent 2's task in isolation — no intake agent, no triage agent.
The structured case is injected directly as the task input.

Run from backend/:
    python test/test_symptom_crew.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from crewai import Crew, Task, Process
from app.crew.agents import make_symptom_agent

# ── Structured case ────────────────────────────────────────────────────────────
CASE = {
    # Profile — strings
    "species":  "dog",
    "breed":    "labrador",
    "sex":      "male",
    "neutered": "yes",

    # Profile — numeric
    "age_years": 3.0,
    "weight_kg": 25.0,

    # Visit history — int
    "vaccinated":            1,
    "num_previous_visits":   0,
    "prev_diagnosis_class":  -1,
    "days_since_last_visit": 0,
    "chronic_flag":          0,

    # Gastro / general — int
    "vomiting":      1,
    "diarrhea":      1,
    "fever":         0,
    "lethargy":      1,
    "loss_appetite": 1,
    "dehydration":   0,

    # Skin — int
    "itching":      0,
    "red_skin":     0,
    "hair_loss":    0,
    "skin_lesions": 0,
    "wounds":       0,

    # Tick fever — int
    "dark_urine":    0,
    "pale_gums":     0,
    "pale_eyelids":  0,
    "tick_exposure": 0,

    # UTI — int
    "pain_urinating":     0,
    "frequent_urination": 0,
    "blood_in_urine":     0,
}

# ── Build agent + single task ──────────────────────────────────────────────────
agent = make_symptom_agent()

task = Task(
    description=(
        "You must use the symptom_classifier tool.\n\n"
        "After the tool returns a result:\n"
        "- Return EXACTLY the JSON from the tool\n"
        "- Do NOT modify any values\n"
        "- Do NOT remove fields\n"
        "- Do NOT add explanations\n"
        "- Do NOT add text before or after\n\n"
        "Final answer must be pure JSON.\n\n"
        f"Case: {json.dumps(CASE)}"
    ),
    expected_output=(
        "Pure JSON exactly as returned by the symptom_classifier tool. "
        "Keys: status, top_prediction, confidence, top3, alternatives, "
        "probability_map, uncertainty_flag, uncertainty_reason, top_gap, possible_out_of_scope. "
        "No extra text, no explanation, no wrapping."
    ),
    agent=agent,
)

crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True,
)

# ── Run ────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("Running Agent 2 CrewAI integration test...")
print("="*60 + "\n")

result = crew.kickoff()

print("\n" + "="*60)
print("CREW RAW OUTPUT (LLM final answer):")
print("="*60)
print(result)

# ── Validate via direct tool call (source of truth) ───────────────────────────
print("\n" + "="*60)
print("DIRECT TOOL VALIDATION:")
print("="*60)
from app.tools.symptom_tools import predict, analyse

tool_result   = predict(CASE)
tool_analysis = analyse(tool_result)

print(f"  top_prediction : {tool_result.top_label}")
print(f"  confidence     : {tool_result.top_confidence:.4f}")
print(f"  uncertainty    : {tool_analysis['uncertainty_flag']}")
print(f"  top3           : {tool_result.top3}")

assert tool_result.top_label == "Parvo",          f"Expected Parvo, got {tool_result.top_label}"
assert tool_result.top_confidence > 0.85,         f"Confidence too low: {tool_result.top_confidence}"
assert tool_analysis["uncertainty_flag"] is False, "Should not be uncertain for clear Parvo case"

print("\nALL ASSERTIONS PASSED -- Agent 2 tool working correctly.")
