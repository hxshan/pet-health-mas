# """
# Intake Agent
# -------------
# Responsibilities:
#   - Parse raw_text_input
#   - Extract pet profile and symptoms
#   - Detect missing info and generate follow-up questions
#   - Populate: pet_profile, extracted_symptoms, structured_case,
#               follow_up_questions, intake_status

# TODO: implement run().
# """
from app.tools.intake_tools.entity_extractor import EntityExtractorTool


def run(state):

    user_input = state.get("raw_text_input", "")

    tool = EntityExtractorTool()

    try:
        data = tool._run(user_input)
    except Exception:
        data = {
            "pet_profile": {},
            "extracted_symptoms": [],
            "structured_case": {},
            "follow_up_questions": ["Please provide more details about the pet."],
            "image_available": False,
        }

    state["pet_profile"] = data.get("pet_profile", {})
    state["extracted_symptoms"] = data.get("extracted_symptoms", [])
    state["structured_case"] = data.get("structured_case", {})
    state["follow_up_questions"] = data.get("follow_up_questions", [])
    state["image_available"] = data.get("image_available", False)

    state["intake_status"] = (
        "complete" if not state["follow_up_questions"] else "needs_more_info"
    )

    return state
    """
    Intake Agent
    """

    user_input = state.get("raw_text_input", "")

    tool = EntityExtractorTool()

    try:
        data = tool._run(user_input)
    except Exception:
        data = {
            "pet_profile": {
                "species": "unknown",
                "breed": "unknown",
                "age": "unknown",
                "sex": "unknown",
                "weight": "unknown",
            },
            "extracted_symptoms": [],
            "structured_case": {},
            "follow_up_questions": ["Could you provide more details about the pet?"],
            "image_available": False,
        }

    # Extract values safely
    pet_profile = data.get("pet_profile", {})
    symptoms = data.get("extracted_symptoms", [])
    follow_up = data.get("follow_up_questions", [])

    # Update global state
    state["pet_profile"] = pet_profile
    state["extracted_symptoms"] = symptoms
    state["structured_case"] = data.get("structured_case", {})
    state["follow_up_questions"] = follow_up
    state["image_available"] = data.get("image_available", False)

    state["intake_status"] = "complete" if not follow_up else "needs_more_info"

    return state