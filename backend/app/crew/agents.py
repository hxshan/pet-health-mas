"""
CrewAI Agent definitions for Pet Health MAS.

Each agent is wired to its tools once the tools are implemented.
Teams: implement your tool in app/tools/<name>/, then uncomment the import
and pass it into your agent's `tools=[...]` list below.
"""
from crewai import Agent

from app.config import settings
# from app.tools.intake_tools import EntityExtractorTool       # uncomment when ready
# from app.tools.symptom_tools import SymptomClassifierTool    # uncomment when ready
# from app.tools.image_tools import ImageClassifierTool        # uncomment when ready
# from app.tools.triage_tools import UrgencyCalculatorTool     # uncomment when ready


def make_intake_agent() -> Agent:
    """
    Intake Agent — parses raw text into a structured case.
    TODO: add tools=[EntityExtractorTool()] once intake_tools is implemented.
    """
    return Agent(
        role="Pet Health Intake Specialist",
        goal=(
            "Parse the owner's description, extract the pet profile and symptoms, "
            "identify missing critical information, and produce a structured case."
        ),
        backstory=(
            "You are an experienced veterinary intake coordinator. "
            "You ask clear follow-up questions when information is missing and "
            "never assume facts not stated by the owner."
        ),
        tools=[],  # TODO: add EntityExtractorTool() here
        llm=f"ollama/{settings.OLLAMA_MODEL}",
        verbose=True,
    )


def make_symptom_agent() -> Agent:
    """
    Symptom Assessment Agent — classifies symptoms using a local ML model.
    TODO: add tools=[SymptomClassifierTool()] once symptom_tools is implemented.
    """
    return Agent(
        role="Veterinary Symptom Analyst",
        goal=(
            "Use the symptom classifier to predict likely conditions, "
            "evaluate confidence, detect uncertainty, and flag possible "
            "out-of-scope cases."
        ),
        backstory=(
            "You are a bounded diagnostic assistant. You only report conditions "
            "supported by the local ML model. When confidence is low you clearly "
            "state 'low confidence result' and recommend veterinary consultation."
        ),
        tools=[],  # TODO: add SymptomClassifierTool() here
        llm=f"ollama/{settings.OLLAMA_MODEL}",
        verbose=True,
    )


def make_image_agent() -> Agent:
    """
    Image Assessment Agent — validates image and runs local image ML model.
    TODO: add tools=[ImageClassifierTool()] once image_tools is implemented.
    """
    return Agent(
        role="Veterinary Image Analyst",
        goal=(
            "Validate the submitted image, run the local image classifier, "
            "assess prediction reliability, and detect uncertainty."
        ),
        backstory=(
            "You are a bounded visual diagnostic assistant. "
            "You never hallucinate conditions not present in the classifier output. "
            "If the image is unusable you report that clearly."
        ),
        tools=[],  # TODO: add ImageClassifierTool() here
        llm=f"ollama/{settings.OLLAMA_MODEL}",
        verbose=True,
    )


def make_triage_agent() -> Agent:
    """
    Triage & Synthesis Agent — combines assessments into a final report.
    TODO: add tools=[UrgencyCalculatorTool()] once triage_tools is implemented.
    """
    return Agent(
        role="Pet Health Triage Coordinator",
        goal=(
            "Combine symptom and image assessments, resolve conflicts, "
            "assign an urgency level, and generate a safe final report."
        ),
        backstory=(
            "You are the final decision layer of a bounded triage system. "
            "You never diagnose definitively. When uncertainty is present "
            "you always recommend veterinary consultation."
        ),
        tools=[],  # TODO: add UrgencyCalculatorTool() here
        llm=f"ollama/{settings.OLLAMA_MODEL}",
        verbose=True,
    )
