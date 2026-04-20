"""
Image Assessment Agent
-----------------------
Responsibilities:
  - Check image_available flag — skip if no image
  - Validate and classify the image at image_path
  - Evaluate confidence and detect uncertainty
  - Populate: image_assessment, uncertainty_flags, possible_out_of_scope

TODO: implement run().
"""
from app.state.schema import PetCaseState


def run(state: PetCaseState) -> PetCaseState:
    raise NotImplementedError
