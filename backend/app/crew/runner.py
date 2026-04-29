# app/crew/runner.py

from app.agents.intake_agent.agent import run as intake_run
from app.crew.crew_setup import build_crew
import json


def run_case(state):

    # 1️⃣ Run Agent 1 ONLY
    state = intake_run(state)

    # 2️⃣ STOP if not enough info
    if state.get("intake_status") != "complete":
        return state

    # 3️⃣ THEN run Agent 2/3/4
    crew = build_crew(image_available=False)

    try:
        result = crew.kickoff(inputs=state)
        parsed = json.loads(str(result))

        state["triage_result"] = parsed
        state["final_report"] = parsed.get("final_report", {})

    except Exception:
        state["final_report"] = {
            "summary": "Pipeline error — veterinary consultation recommended."
        }

    return state