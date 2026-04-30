"""
Symptom Assessment Agent
--------------------------
Responsibilities:
  - Use extracted_symptoms from state
  - Run the symptom classifier tool
  - Evaluate confidence and detect uncertainty
  - Populate: symptom_assessment, uncertainty_flags, possible_out_of_scope
"""
from app.agents.symptom_agent.logic import run_symptom_assessment
from app.state.schema import PetCaseState


def run(state: PetCaseState) -> PetCaseState:
    """
    Direct (non-CrewAI) runner for the Symptom Assessment Agent.
    Merges structured_case + top-level state fields so all profile
    and symptom flag data is available to the logic layer.
    """
    # Merge structured_case into state so logic.py sees all fields in one flat dict
    case = {**state, **state.get("structured_case", {})}

    result = run_symptom_assessment(case)
    state["symptom_assessment"] = result

    if result.get("uncertainty_flag"):
        state.setdefault("uncertainty_flags", []).append("symptom_low_confidence")
    if result.get("possible_out_of_scope"):
        state["possible_out_of_scope"] = True

    return state
