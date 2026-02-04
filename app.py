import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import xgboost as xgb
import os

app = Flask(__name__)
CORS(app)

# Load preprocessors
pre_motor = joblib.load("models/pre_motor.pkl")
pre_total = joblib.load("models/pre_total.pkl")

# Load models
model_motor = xgb.XGBRegressor()
model_motor.load_model("models/motor.json")

model_total = xgb.XGBRegressor()
model_total.load_model("models/total.json")

EXPECTED_FEATURES = [
    'age','sex','test_time',
    'Jitter(%)','Jitter:PPQ5',
    'Shimmer(dB)','Shimmer:APQ5',
    'NHR','HNR','RPDE','DFA','PPE'
]

@app.route('/', methods=['GET'])
def index():
    return "Server is running"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        missing = [f for f in EXPECTED_FEATURES if f not in data]
        if missing:
            return jsonify({"error": f"Missing: {missing}"}), 400

        df = pd.DataFrame([data], columns=EXPECTED_FEATURES)

        df_motor = pre_motor.transform(df)
        df_total = pre_total.transform(df)

        motor_pred = float(model_motor.predict(df_motor)[0])
        total_pred = float(model_total.predict(df_total)[0])

        return jsonify({
            "motor_UPDRS": motor_pred,
            "total_UPDRS": total_pred
        })

    except Exception as e:
        print("Flask error:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
