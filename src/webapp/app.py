import os, json
from flask import Flask, request, jsonify, render_template, abort
import joblib
import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model_rf.joblib")

app = Flask(__name__)

# Load model at startup (if present)
MODEL_OBJ = None
FEATURE_NAMES = None
if os.path.exists(MODEL_PATH):
    OBJ = joblib.load(MODEL_PATH)
    MODEL_OBJ = OBJ.get("pipeline", OBJ)
    FEATURE_NAMES = OBJ.get("feature_names")

@app.route("/")
def index():
    has_model = MODEL_OBJ is not None
    return render_template("index.html", has_model=has_model)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "has_model": MODEL_OBJ is not None})

def _predict_from_row(numbers, threshold=0.5):
    """Make prediction from EEG data row"""
    try:
        if MODEL_OBJ is None:
            return None, "Model not loaded. Place model_rf.joblib next to app.py and restart."

        # Validate input range (EEG values are typically ±200-300μV)
        if any(abs(x) > 5000 for x in numbers):
            return None, "EEG values outside expected range (±5000)"

        arr = np.array(numbers, dtype=float).reshape(1, -1)
        if FEATURE_NAMES is not None and len(FEATURE_NAMES) == arr.shape[1]:
            X = pd.DataFrame(arr, columns=FEATURE_NAMES)
        else:
            X = pd.DataFrame(arr)

        if hasattr(MODEL_OBJ, "predict_proba"):
            prob = float(MODEL_OBJ.predict_proba(X)[0, 1])
        elif hasattr(MODEL_OBJ, "decision_function"):
            score = float(MODEL_OBJ.decision_function(X)[0])
            prob = 1.0 / (1.0 + np.exp(-score))
        else:
            label = int(MODEL_OBJ.predict(X)[0])
            return {"pred": label, "label": "Seizure" if label == 1 else "Non-seizure",
                    "prob_seizure": None, "threshold": threshold}, None

        label = 1 if prob >= threshold else 0
        return {"pred": label, "label": "Seizure" if label == 1 else "Non-seizure",
                "prob_seizure": prob, "threshold": threshold}, None

    except Exception as e:
        return None, f"Prediction error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row = data.get("row")
    threshold = data.get("threshold", 0.424)
    try:
        threshold = float(threshold)
    except Exception:
        return jsonify({"error": "Invalid threshold."}), 400

    if not isinstance(row, list):
        return jsonify({"error": "Expected JSON with 'row': [178 numbers]."}), 400

    row = [r for r in row if str(r).strip() != ""]

    try:
        nums = [float(x) for x in row]
    except Exception:
        return jsonify({"error": "Row must be numeric values."}), 400
    
    if len(nums) > 178:
        nums = nums[:178]

    if len(nums) < 178:
        return jsonify({"error": f"Expected 178 numbers, got {len(nums)}."}), 400

    is_valid, message = validate_eeg_input(nums)
    if not is_valid:
        return jsonify({"error": message}), 400

    out, err = _predict_from_row(nums, threshold)
    if err:
        return jsonify({"error": err}), 500
    
    return jsonify(out)


# helper function
def validate_eeg_input(numbers):
    if len(numbers) != 178:
        return False, f"Expected 178 values, got {len(numbers)}"

    try:
        [float(x) for x in numbers]
    except ValueError:
        return False, "All values must be numeric"

    return True, "OK"

@app.route("/model-info")
def model_info():
    if MODEL_OBJ is None:
        return jsonify({"error": "No model loaded"})

    info = {
        "model_type": type(MODEL_OBJ).__name__,
        "features_expected": 178,
        "threshold_default": 0.5
    }
    return jsonify(info)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
