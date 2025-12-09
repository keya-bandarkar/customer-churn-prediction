import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "telco_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")
