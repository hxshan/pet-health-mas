"""
Intake Agent
-------------
Responsibilities:
  - Parse raw_text_input
  - Extract pet profile and symptoms
  - Detect missing info and generate follow-up questions
  - Populate: pet_profile, extracted_symptoms, structured_case,
              follow_up_questions, intake_status

TODO: implement run().
"""
from app.state.schema import PetCaseState


def run(state: PetCaseState) -> PetCaseState:
    raise NotImplementedError
