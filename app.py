import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the two models
# model_motor = joblib.load("models/park_xgb_motor_updrs.pkl")
# model_total = joblib.load("models/park_xgb_total_updrs.pkl")
model_motor = joblib.load("models/park_xgb_motor_updrs_cpu.pkl")
model_total = joblib.load("models/park_xgb_total_updrs_cpu.pkl")

EXPECTED_FEATURES = [
    'age', 'sex', 'test_time',
    'Jitter(%)', 'Jitter:PPQ5',
    'Shimmer(dB)', 'Shimmer:APQ5',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

@app.route('/', methods=['GET'])
def index():
    return "Server is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame with headers
        df = pd.DataFrame([data], columns=EXPECTED_FEATURES)

        # Predict using both models
        motor_UPDRS = float(model_motor.predict(df)[0])
        total_UPDRS = float(model_total.predict(df)[0])

        return jsonify({
            "motor_UPDRS": motor_UPDRS,
            "total_UPDRS": total_UPDRS
        })

    except Exception as e:
        print("Flask error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)