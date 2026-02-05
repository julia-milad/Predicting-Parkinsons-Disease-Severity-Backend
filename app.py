import os
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from train import train_and_save_models  # Import the training function

app = Flask(__name__)

# Absolute paths for models
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TOTAL_MODEL_PATH = os.path.join(MODEL_DIR, "total_UPDRS.pkl")
MOTOR_MODEL_PATH = os.path.join(MODEL_DIR, "motor_UPDRS.pkl")

# Ensure models exist, else train them
if not (os.path.exists(TOTAL_MODEL_PATH) and os.path.exists(MOTOR_MODEL_PATH)):
    print("Models not found. Training now...")
    train_and_save_models(data_path="data/parkinsons_updrs.data", model_dir=MODEL_DIR)
    print("Training complete!")

# Load pre-trained models
total_model = joblib.load(TOTAL_MODEL_PATH)
motor_model = joblib.load(MOTOR_MODEL_PATH)

# Expected input features
FEATURES = [
    'age','sex','test_time',
    'Jitter(%)','Jitter:PPQ5',
    'Shimmer(dB)','Shimmer:APQ5',
    'NHR','HNR','RPDE','DFA','PPE'
]

@app.route("/", methods=["GET"])
def index():
    return "Server is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Validate input features
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert to DataFrame
        X = pd.DataFrame([data], columns=FEATURES)

        # Make predictions
        total_pred = float(total_model.predict(X)[0])
        motor_pred = float(motor_model.predict(X)[0])

        return jsonify({
            "motor_UPDRS": max(0, motor_pred),
            "total_UPDRS": max(0, total_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Only used for local testing; in production, use Gunicorn
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)