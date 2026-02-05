import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def train_and_save_models(data_path="data/parkinsons_updrs.data", model_dir="models"):
    # Ensure model directory exists
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    data = pd.read_csv(data_path)
    
    # Drop unnecessary columns
    drop_columns = [
        'subject#',
        'Jitter(Abs)', 'Jitter:RAP', 'Jitter:DDP',
        'Shimmer', 'Shimmer:APQ3', 'Shimmer:APQ11', 'Shimmer:DDA'
    ]
    data = data.drop(columns=[c for c in drop_columns if c in data.columns])
    
    FEATURES = [
        'age','sex','test_time',
        'Jitter(%)','Jitter:PPQ5',
        'Shimmer(dB)','Shimmer:APQ5',
        'NHR','HNR','RPDE','DFA','PPE'
    ]
    
    X = data[FEATURES]
    y_total = data['total_UPDRS']
    y_motor = data['motor_UPDRS']
    
    # Preprocessing
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, FEATURES)
    ])
    
    def build_model():
        return Pipeline([
            ('pre', preprocessor),
            ('xgb', xgb.XGBRegressor(
                objective='reg:squarederror',
                tree_method='hist',
                predictor='cpu_predictor',
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=42
            ))
        ])
    
    # Train models
    total_model = build_model().fit(X, y_total)
    motor_model = build_model().fit(X, y_motor)
    
    # Save models
    joblib.dump(total_model, os.path.join(model_dir, "total_UPDRS.pkl"))
    joblib.dump(motor_model, os.path.join(model_dir, "motor_UPDRS.pkl"))
    
    print("Models exported successfully.")
    
    # Optional: print metrics
    pred_total = total_model.predict(X)
    pred_motor = motor_model.predict(X)
    print("Total R2:", r2_score(y_total, pred_total))
    print("Total RMSE:", np.sqrt(mean_squared_error(y_total, pred_total)))
    print("Motor R2:", r2_score(y_motor, pred_motor))
    print("Motor RMSE:", np.sqrt(mean_squared_error(y_motor, pred_motor)))
    
    return total_model, motor_model


# If you want to train directly when running this file
if __name__ == "__main__":
    train_and_save_models()