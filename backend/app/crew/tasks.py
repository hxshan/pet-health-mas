"""
CrewAI Task definitions for Pet Health MAS.

Tasks are created fresh on every run (they capture agent + context references).
Call `build_tasks(...)` from crew_setup.py.
"""
from typing import Optional

from crewai import Agent, Task

from app.agents.symptom_agent.prompt import CONSTRAINTS as SYMPTOM_CONSTRAINTS


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
            "Parse the owner's raw text input. "
            "Extract: pet profile (species, breed, age, sex, weight), "
            "reported symptoms, duration/severity hints. "
            "Identify any missing critical information. "
            "Output a structured JSON case."
        ),
        expected_output=(
            "JSON with keys: pet_profile, extracted_symptoms, structured_case, "
            "follow_up_questions, image_available."
        ),
        agent=intake_agent,
    )

    # ------------------------------------------------------------------
    # Task 2 — Symptom Assessment
    # ------------------------------------------------------------------
    symptom_task = Task(
        description=(
            "You are the Symptom Assessment Agent.\n"
            "A structured pet case has been prepared. "
            "Call the symptom_classifier tool once with the pet profile and "
            "symptom flags from the case. "
            "Then return the tool output directly as your Final Answer — "
            "do not add extra reasoning.\n\n"
            f"{SYMPTOM_CONSTRAINTS}"
        ),
        expected_output=(
            "A JSON object with keys: assessment_status, top_prediction, "
            "confidence, alternatives, uncertainty_flag, uncertainty_reason, "
            "possible_out_of_scope, needs_more_info, missing_fields, "
            "local_interpretation."
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
