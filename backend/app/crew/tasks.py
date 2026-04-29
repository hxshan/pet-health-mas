"""
CrewAI Task definitions for Pet Health MAS.

Tasks are created fresh on every run.
Agent 1 (Intake) is handled OUTSIDE Crew.
Crew only runs Agent 2, 3, 4.
"""

from crewai import Task


def build_tasks(
    intake_agent,   # NOT USED (kept for compatibility)
    symptom_agent,
    image_agent,
    triage_agent,
    image_available: bool = False,
) -> list[Task]:
    """
    Build tasks ONLY for:
    - Agent 2 (Symptom)
    - Agent 3 (Image - optional)
    - Agent 4 (Triage)
    """

    # --------------------------------------------------
    # Task 1 — Symptom Assessment (Agent 2)
    # --------------------------------------------------
    symptom_task = Task(
        description=(
            "You receive extracted symptoms from Agent 1.\n\n"

            "Your job:\n"
            "- Analyze symptoms\n"
            "- Run symptom classifier tool\n"
            "- Identify most likely condition\n"
            "- Provide confidence score\n"
            "- Suggest alternatives\n\n"

            "Rules:\n"
            "- DO NOT hallucinate conditions\n"
            "- Stay within model scope\n"
            "- Return ONLY valid JSON"
        ),
        expected_output=(
            "{\n"
            '  "top_prediction": "",\n'
            '  "confidence": 0.0,\n'
            '  "alternatives": [],\n'
            '  "uncertainty_flag": false,\n'
            '  "possible_out_of_scope": false,\n'
            '  "local_interpretation": ""\n'
            "}"
        ),
        agent=symptom_agent,
    )

    # --------------------------------------------------
    # Task 2 — Image Assessment (Agent 3)
    # --------------------------------------------------
    image_task = Task(
        description=(
            "You receive an image and case context.\n\n"

            "Your job:\n"
            "- Validate image quality\n"
            "- Check if image is relevant\n"
            "- Run image classifier\n"
            "- Evaluate prediction confidence\n\n"

            "Rules:\n"
            "- If image unusable → set image_validity='unusable'\n"
            "- DO NOT hallucinate\n"
            "- Return ONLY valid JSON"
        ),
        expected_output=(
            "{\n"
            '  "image_prediction": "",\n'
            '  "confidence": 0.0,\n'
            '  "alternatives": [],\n'
            '  "uncertainty_flag": false,\n'
            '  "possible_out_of_scope": false,\n'
            '  "image_validity": "valid|unusable",\n'
            '  "local_interpretation": ""\n'
            "}"
        ),
        agent=image_agent,
    )

    # --------------------------------------------------
    # Task 3 — Triage & Synthesis (Agent 4)
    # --------------------------------------------------
    triage_context = [symptom_task]

    if image_available:
        triage_context.append(image_task)

    triage_task = Task(
        description=(
            "Combine results from symptom and image analysis.\n\n"

            "Your job:\n"
            "- Resolve agreement or conflict\n"
            "- Detect uncertainty\n"
            "- Assign urgency level\n"
            "- Generate final report\n\n"

            "Urgency levels:\n"
            "- non-urgent\n"
            "- monitor\n"
            "- vet_soon\n"
            "- urgent\n\n"

            "Rules:\n"
            "- NEVER give definitive diagnosis\n"
            "- Recommend vet if uncertain\n"
            "- Return ONLY valid JSON"
        ),
        expected_output=(
            "{\n"
            '  "primary_assessment": "",\n'
            '  "urgency_level": "",\n'
            '  "recommendation": "",\n'
            '  "uncertainty_status": "",\n'
            '  "final_report": {}\n'
            "}"
        ),
        agent=triage_agent,
        context=triage_context,
    )

    # --------------------------------------------------
    # FINAL TASK ORDER
    # --------------------------------------------------
    tasks = [symptom_task, triage_task]

    if image_available:
        tasks = [symptom_task, image_task, triage_task]

    return tasks