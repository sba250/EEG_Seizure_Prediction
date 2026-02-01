# Seizure Predictor — Mini Web App

Tiny Flask app that loads your `model_rf.joblib` and serves a web page where you paste 178 numbers to get:
- predicted label (Seizure / Non-seizure)
- probability of seizure
- threshold used (default 0.424, editable in the UI)

## Quick start
1) Put `model_rf.joblib` next to `app.py`.
2) Create a venv and install deps:
```bash
cd seizure_webapp
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
3) Run the app:
```bash
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --port 5000
# open http://127.0.0.1:5000
```
Or: `python app.py`.
