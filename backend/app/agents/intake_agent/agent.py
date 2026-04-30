from app.tools.intake_tools.entity_extractor import EntityExtractorTool
from app.llm.ollama_client import chat_json


# ----------------------------------------
# Normalize symptoms (for Agent 2 ML)
# ----------------------------------------
def normalize_symptoms(symptoms):
    mapping = {
        "not eating": "loss_of_appetite",
        "no appetite": "loss_of_appetite",
        "not eating anything": "loss_of_appetite",
        "throwing up": "vomiting",
        "has vomiting": "vomiting",
    }
    return [mapping.get(s.lower(), s.lower()) for s in symptoms]


# ----------------------------------------
# Ask ONLY ONE follow-up question
# ----------------------------------------
def generate_followup(structured_case, answered_questions):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a veterinary intake assistant.\n"
                "Ask ONLY ONE most important missing question.\n"
                "Do NOT repeat or rephrase previous questions.\n"
                "Be short.\n"
                "Return ONLY JSON."
            ),
        },
        {
            "role": "user",
            "content": f"""
Case:
{structured_case}

Already asked:
{list(answered_questions.keys())}

Missing:
- duration
- appetite
- vomiting/diarrhea

Return ONLY:
{{
  "question": "one short question"
}}
""",
        },
    ]

    result = chat_json(messages)

    if not isinstance(result, dict):
        return None

    return result.get("question")


# ----------------------------------------
# Helper: check if info already collected
# ----------------------------------------
def has_answer(answers, keywords):
    return any(
        any(k in q.lower() for k in keywords)
        for q in answers
    )


# ----------------------------------------
# MAIN AGENT RUN
# ----------------------------------------
def run(state):
    tool = EntityExtractorTool()

    raw_input = state.get("raw_text_input", "")
    answers = state.get("follow_up_answers", {})

    # ----------------------------------------
    # Merge conversation (LIMIT context)
    # ----------------------------------------
    merged_input = raw_input
    for q, a in list(answers.items())[-2:]:
        merged_input += f"\n{q}: {a}"

    # ----------------------------------------
    # Extract using LLM Tool
    # ----------------------------------------
    try:
        data = tool._run(merged_input)
    except Exception:
        data = {
            "species": "unknown",
            "breed": "unknown",
            "age": "unknown",
            "sex": "unknown",
            "weight": "unknown",
            "raw_symptoms": [],
        }

    # ----------------------------------------
    # Normalize symptoms
    # ----------------------------------------
    symptoms = normalize_symptoms(data.get("raw_symptoms", []))

    # ----------------------------------------
    # STRONG STOP CONDITION (CRITICAL)
    # ----------------------------------------
    enough_info = (
        len(symptoms) > 0 and
        has_answer(answers, ["duration", "how long"]) and
        has_answer(answers, ["appetite", "eating"]) and
        has_answer(answers, ["vomit", "vomiting", "diarrhea"])
    )

    # ----------------------------------------
    # Ask next question
    # ----------------------------------------
    question = None

    if not enough_info:
        question = generate_followup(data, answers)

        # HARD BLOCK duplicates
        if question:
            q_lower = question.lower()

            if "duration" in q_lower and has_answer(answers, ["duration", "how long"]):
                question = None

            if "appetite" in q_lower and has_answer(answers, ["appetite", "eating"]):
                question = None

            if "vomit" in q_lower and has_answer(answers, ["vomit", "vomiting"]):
                question = None

    # ----------------------------------------
    # Final decision
    # ----------------------------------------
    if question:
        state["follow_up_questions"] = [question]
        state["intake_status"] = "needs_more_info"
    else:
        state["follow_up_questions"] = []
        state["intake_status"] = "complete"

    # ----------------------------------------
    # Structured output (VERY IMPORTANT)
    # ----------------------------------------
    state["pet_profile"] = {
        "species": data.get("species", "unknown"),
        "breed": data.get("breed", "unknown"),
        "age": data.get("age", "unknown"),
        "sex": data.get("sex", "unknown"),
        "weight": data.get("weight", "unknown"),
    }

    state["extracted_symptoms"] = symptoms

    state["structured_case"] = {
        "pet_profile": state["pet_profile"],
        "symptoms": symptoms,
        "answers": answers,
    }

    # ----------------------------------------
    # FLAG FOR NEXT AGENTS
    # ----------------------------------------
    state["data_sufficient"] = state["intake_status"] == "complete"

    return state