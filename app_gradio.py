import gradio as gr
import pandas as pd
import joblib
from src.config import MODEL_PATH, PREPROCESSOR_PATH

# Load artifacts once
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_churn(
    gender,
    senior_citizen,
    partner,
    dependents,
    tenure,
    phone_service,
    multiple_lines,
    internet_service,
    online_security,
    online_backup,
    device_protection,
    tech_support,
    streaming_tv,
    streaming_movies,
    contract,
    paperless_billing,
    payment_method,
    monthly_charges,
    total_charges,
):
    # Build feature dict (match Telco dataset column names)
    customer_dict = {
        "gender": gender,
        "SeniorCitizen": int(senior_citizen),
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df = pd.DataFrame([customer_dict])
    X_prep = preprocessor.transform(df)
    prob_churn = model.predict_proba(X_prep)[0, 1]
    prob_percent = round(prob_churn * 100, 2)

    if prob_churn >= 0.7:
        risk = "High churn risk"
    elif prob_churn >= 0.4:
        risk = "Medium churn risk"
    else:
        risk = "Low churn risk"

    return f"{prob_percent} %", risk


with gr.Blocks(title="Customer Churn Prediction") as demo:
    gr.Markdown("## 📊 Customer Churn Prediction")
    gr.Markdown("Enter customer details to estimate churn probability.")

    with gr.Row():
        with gr.Column():
            gender = gr.Dropdown(["Female", "Male"], label="Gender", value="Female")
            senior_citizen = gr.Dropdown([0, 1], label="Senior Citizen (0 = No, 1 = Yes)", value=0)
            partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
            phone_service = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            multiple_lines = gr.Dropdown(["No", "Yes", "No phone service"], label="Multiple Lines", value="No")

        with gr.Column():
            internet_service = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
            online_security = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
            online_backup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            device_protection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
            tech_support = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
            streaming_tv = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
            streaming_movies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")

    contract = gr.Dropdown(
        ["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"
    )
    paperless_billing = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
    payment_method = gr.Dropdown(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        label="Payment Method",
        value="Electronic check",
    )
    monthly_charges = gr.Slider(0, 200, value=75, step=1, label="Monthly Charges")
    total_charges = gr.Slider(0, 10000, value=900, step=10, label="Total Charges")

    btn = gr.Button("Predict Churn")

    churn_prob = gr.Textbox(label="Churn Probability", interactive=False)
    churn_risk = gr.Textbox(label="Risk Level", interactive=False)

    inputs = [
        gender,
        senior_citizen,
        partner,
        dependents,
        tenure,
        phone_service,
        multiple_lines,
        internet_service,
        online_security,
        online_backup,
        device_protection,
        tech_support,
        streaming_tv,
        streaming_movies,
        contract,
        paperless_billing,
        payment_method,
        monthly_charges,
        total_charges,
    ]

    btn.click(fn=predict_churn, inputs=inputs, outputs=[churn_prob, churn_risk])


if __name__ == "__main__":
    demo.launch()
