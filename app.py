import joblib
import pandas as pd
import numpy as np
from .config import MODEL_PATH, PREPROCESSOR_PATH

def load_artifacts():
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def predict_churn(customer_features: dict) -> float:
    """
    customer_features: dict with keys matching original dataset columns
    (except customerID and Churn)
    Returns churn probability between 0 and 1.
    """
    model, preprocessor = load_artifacts()

    df = pd.DataFrame([customer_features])
    X_prep = preprocessor.transform(df)
    prob = model.predict_proba(X_prep)[0, 1]
    return float(prob)

if __name__ == "__main__":
    # TODO: fill with real values once you inspect your CSV columns
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.0,
        "TotalCharges": 960.0,
    }

    prob = predict_churn(sample_customer)
    print(f"Churn probability: {prob:.3f}")
