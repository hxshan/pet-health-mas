"""
CrewAI Task definitions for Pet Health MAS.

Tasks are created fresh on every run (they capture agent + context references).
Call `build_tasks(...)` from crew_setup.py.
"""
from typing import Optional

from crewai import Agent, Task


def build_tasks(
    intake_agent: Agent,
    symptom_agent: Agent,
    image_agent: Agent,
    triage_agent: Agent,
    image_available: bool = False,
) -> list[Task]:
    """
    Build and return the ordered task list.

    Task 3 (image assessment) is only added when image_available=True.
    """

    # ------------------------------------------------------------------
    # Task 1 — Intake
    # ------------------------------------------------------------------
    intake_task = Task(
          description=(
            "Parse the owner's raw text input.\n\n"

            "STRICT RULES:\n"
            "- You MUST return ONLY valid JSON\n"
            "- DO NOT explain anything\n"
            "- DO NOT include code\n"
            "- DO NOT include analysis\n"
            "- DO NOT include markdown\n\n"

            "Extract:\n"
            "- pet_profile (species, breed, age, sex, weight)\n"
            "- extracted_symptoms (list)\n"
            "- structured_case\n"
            "- follow_up_questions (list)\n"
            "- image_available (true/false)\n\n"

            "If any field is missing → use 'unknown'\n"
        ),
        expected_output=(
            "ONLY JSON:\n"
            "{\n"
            '  "pet_profile": {},\n'
            '  "extracted_symptoms": [],\n'
            '  "structured_case": {},\n'
            '  "follow_up_questions": [],\n'
            '  "image_available": false\n'
            "}"
        ),
        agent=intake_agent,
    )


    # ------------------------------------------------------------------
    # Task 2 — Symptom Assessment
    # ------------------------------------------------------------------
    symptom_task = Task(
        description=(
            "Using the structured case from the Intake Agent, run the symptom "
            "classifier tool on the extracted_symptoms list. "
            "Evaluate the top prediction and confidence. "
            "Flag uncertainty if confidence is below threshold or if top-2 "
            "predictions are very close. "
            "Do NOT invent conditions outside the classifier scope."
        ),
        expected_output=(
            "JSON with keys: top_prediction, confidence, alternatives, "
            "uncertainty_flag, possible_out_of_scope, local_interpretation."
        ),
        agent=symptom_agent,
        context=[intake_task],
    )

    # ------------------------------------------------------------------
    # Task 3 — Image Assessment (conditional)
    # ------------------------------------------------------------------
    image_task = Task(
        description=(
            "Validate the submitted image and run the image classifier tool. "
            "Assess whether the image is usable and relevant. "
            "Evaluate prediction reliability. "
            "If the image is invalid or unusable, set image_validity='unusable'. "
            "Do NOT hallucinate new conditions."
        ),
        expected_output=(
            "JSON with keys: image_prediction, confidence, alternatives, "
            "uncertainty_flag, possible_out_of_scope, image_validity, "
            "local_interpretation."
        ),
        agent=image_agent,
        context=[intake_task],
    )

    # ------------------------------------------------------------------
    # Task 4 — Triage & Synthesis
    # ------------------------------------------------------------------
    triage_context = [symptom_task, image_task] if image_available else [symptom_task]

    triage_task = Task(
        description=(
            "Combine the symptom assessment (and image assessment if present). "
            "Resolve agreement, conflict, or coexisting conditions. "
            "Assign an urgency level: non-urgent | monitor | vet_soon | urgent. "
            "Generate a final report. "
            "If either assessment has uncertainty_flag=True, recommend veterinary "
            "consultation. Never claim to definitively diagnose any condition."
        ),
        expected_output=(
            "JSON with keys: primary_assessment, urgency_level, recommendation, "
            "uncertainty_status, final_report."
        ),
        agent=triage_agent,
        context=triage_context,
    )

    tasks = [intake_task, symptom_task, triage_task]
    if image_available:
        # Insert image task before triage
        tasks = [intake_task, symptom_task, image_task, triage_task]

    return tasks
