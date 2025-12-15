import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model_motor = joblib.load("models/park_xgb_motor_updrs.pkl")
model_total = joblib.load("models/park_xgb_total_updrs.pkl")

EXPECTED_FEATURES = [
    'age', 'sex', 'test_time',
    'Jitter(%)', 'Jitter:PPQ5',
    'Shimmer(dB)', 'Shimmer:APQ5',
    'NHR', 'HNR', 'RPDE', 'DFA', 'PPE'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        for feat in EXPECTED_FEATURES:
            if feat not in data:
                return jsonify({"error": f"Missing feature: {feat}"}), 400

        df = pd.DataFrame([data], columns=EXPECTED_FEATURES)
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
