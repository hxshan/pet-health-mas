"""
Triage & Synthesis Agent
--------------------------
Responsibilities:
  - Combine symptom_assessment and image_assessment from state
  - Resolve agreement / conflict between the two
  - Assign urgency level: non-urgent | monitor | vet_soon | urgent
  - Generate final report with disclaimer
  - Populate: triage_result, final_report

TODO: implement run().
"""
from app.state.schema import PetCaseState


def run(state: PetCaseState) -> PetCaseState:
    raise NotImplementedError
