cd backend
py -3.11 -m venv venv311
venv311\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload