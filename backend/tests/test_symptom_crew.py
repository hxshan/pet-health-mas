"""
Integration test — Agent 2 (Symptom Assessment) in CrewAI.

Runs a minimal crew with only the Symptom Assessment Agent and its task,
bypassing Agents 1, 3, and 4 so you can test Agent 2 in isolation.

Usage (from backend/ folder):
    python tests/test_symptom_crew.py

Prerequisites:
    - Ollama running:  ollama serve
    - Model pulled:    ollama pull llama3.1:8b
    - venv active with requirements installed
"""
import json
import sys
import os

# Make sure imports resolve from backend/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crewai import Crew, Process, Task
from app.crew.agents import make_symptom_agent
from app.agents.symptom_agent.prompt import CONSTRAINTS

# ── Sample structured case (normally produced by Agent 1) ─────────────────────
SAMPLE_CASE = {
    # Profile — strings
    "species":  "dog",
    "breed":    "labrador",
    "sex":      "male",
    "neutered": "yes",

    # Profile — numeric
    "age_years": 3.0,
    "weight_kg": 28.0,

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


def run_agent2_crew_test():
    print("\n" + "="*60)
    print("  Agent 2 — Symptom Assessment CrewAI Integration Test")
    print("="*60)

    symptom_agent = make_symptom_agent()

    # Standalone task — no intake context needed for this isolated test
    task = Task(
        description=(
            f"Assess the following structured pet case and return a symptom assessment.\n\n"
            f"Case:\n{json.dumps(SAMPLE_CASE, indent=2)}\n\n"
            f"{CONSTRAINTS}"
        ),
        expected_output=(
            "A JSON object with keys: assessment_status, top_prediction, "
            "confidence, alternatives, uncertainty_flag, uncertainty_reason, "
            "possible_out_of_scope, needs_more_info, missing_fields, local_interpretation."
        ),
        agent=symptom_agent,
    )

    crew = Crew(
        agents=[symptom_agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
    )

    print("\n▶  Kicking off crew...\n")
    result = crew.kickoff()

    print("\n" + "="*60)
    print("  Raw crew output:")
    print("="*60)
    print(result)

    # Try to parse as JSON
    try:
        parsed = json.loads(str(result))
        print("\n  Parsed JSON output:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        print("\n  (Output is not pure JSON — check agent prompt if needed)")

    print("\n✅  Test complete.")


if __name__ == "__main__":
    run_agent2_crew_test()
