"""
Symptom Assessment Agent
--------------------------
Responsibilities:
  - Use extracted_symptoms from state
  - Run the symptom classifier tool
  - Evaluate confidence and detect uncertainty
  - Populate: symptom_assessment, uncertainty_flags, possible_out_of_scope

TODO: implement run().
"""
from app.state.schema import PetCaseState


def run(state: PetCaseState) -> PetCaseState:
    raise NotImplementedError
