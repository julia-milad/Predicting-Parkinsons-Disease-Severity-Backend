import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load pipelines
total_model = joblib.load("models/total_UPDRS.pkl")
motor_model = joblib.load("models/motor_UPDRS.pkl")

FEATURES = [
    'age','sex','test_time',
    'Jitter(%)','Jitter:PPQ5',
    'Shimmer(dB)','Shimmer:APQ5',
    'NHR','HNR','RPDE','DFA','PPE'
]


@app.route('/', methods=['GET'])
def index():
    return "Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        X = pd.DataFrame([data], columns=FEATURES)

        total_pred = float(total_model.predict(X)[0])
        motor_pred = float(motor_model.predict(X)[0])

        return jsonify({
            "motor_UPDRS": max(0, motor_pred),
            "total_UPDRS": max(0, total_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)