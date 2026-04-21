import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.agents.symptom_agent.logic import run_symptom_assessment

sample_case = {
    "species": "dog",
    "breed": "labrador",
    "sex": "male",
    "neutered": "yes",
    "age_years": 3.0,
    "weight_kg": 25.0,
    "vaccinated": 1,

    "num_previous_visits": 0,
    "prev_diagnosis_class": -1,
    "days_since_last_visit": 0,
    "chronic_flag": 0,

    "vomiting": 1,
    "diarrhea": 1,
    "dehydration": 0,
    "loss_appetite": 1,
    "fever": 0,
    "lethargy": 1,

    "itching": 0,
    "red_skin": 0,
    "hair_loss": 0,
    "skin_lesions": 0,
    "wounds": 0,

    "dark_urine": 0,
    "pale_gums": 0,
    "pale_eyelids": 0,
    "tick_exposure": 0,

    "pain_urinating": 0,
    "frequent_urination": 0,
    "blood_in_urine": 0,
}

result = run_symptom_assessment(sample_case)
print(result)