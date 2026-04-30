from crewai import Crew, Process

from app.crew.agents import (
    make_symptom_agent,
    make_image_agent,
    make_triage_agent,
)
from app.crew.tasks import build_tasks


def build_crew(image_available: bool = False) -> Crew:
    """
    Build Crew ONLY for Agent 2, 3, 4
    Agent 1 runs separately
    """

    symptom_agent = make_symptom_agent()
    image_agent   = make_image_agent()
    triage_agent  = make_triage_agent()

    # ⚠️ intake_agent = None (not used in crew)
    tasks = build_tasks(
        intake_agent=None,  # not used
        symptom_agent=symptom_agent,
        image_agent=image_agent,
        triage_agent=triage_agent,
        image_available=image_available,
    )

    agents = [symptom_agent, triage_agent]

    if image_available:
        agents = [symptom_agent, image_agent, triage_agent]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=True,
    )