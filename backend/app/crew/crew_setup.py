"""
Crew factory for Pet Health MAS.

Usage:
    crew = build_crew(image_available=False)
    result = crew.kickoff(inputs={"raw_text_input": "..."})
"""
from crewai import Crew, Process

from app.crew.agents import (
    make_intake_agent,
    make_symptom_agent,
    make_image_agent,
    make_triage_agent,
)
from app.crew.tasks import build_tasks


def build_crew(image_available: bool = False) -> Crew:
    """
    Instantiate all agents and tasks then assemble the Crew.

    Args:
        image_available: Whether the case includes an image.
                         Controls whether the Image Assessment task is included.

    Returns:
        A configured CrewAI Crew ready to kickoff.
    """
    intake_agent   = make_intake_agent()
    symptom_agent  = make_symptom_agent()
    image_agent    = make_image_agent()
    triage_agent   = make_triage_agent()

    tasks = build_tasks(
        intake_agent=intake_agent,
        symptom_agent=symptom_agent,
        image_agent=image_agent,
        triage_agent=triage_agent,
        image_available=image_available,
    )

    agents = [intake_agent, symptom_agent, triage_agent]
    if image_available:
        agents = [intake_agent, symptom_agent, image_agent, triage_agent]

    return Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,   # agents run in order
        verbose=True,
    )
