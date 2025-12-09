from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import joblib

from .data_loading import load_raw_data
from .preprocessing import prepare_data_for_model
from .config import MODEL_PATH, PREPROCESSOR_PATH


def train_and_save_model():
    df = load_raw_data()
    X_train, X_test, y_train, y_test, preprocessor = prepare_data_for_model(df)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob))

    # Save model + preprocessor
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved preprocessor to {PREPROCESSOR_PATH}")

if __name__ == "__main__":
    train_and_save_model()
