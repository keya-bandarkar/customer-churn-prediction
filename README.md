# Customer Churn Prediction System

This project is a machine learning–based customer churn prediction system designed for the telecom industry. It predicts whether a customer is likely to stop using the service based on demographic information, service subscriptions, and billing details. The project includes an end-to-end data science pipeline along with an interactive Streamlit-based web interface for real-time prediction.

---

## 🔹 Problem Statement

Customer churn is a major business challenge in the telecom sector. Predicting churn in advance helps organizations take preventive actions such as offering discounts, improving services, or personalized retention strategies.

This project aims to:
- Identify customers who are at high risk of churn
- Provide real-time churn probability
- Assist in data-driven business decision making

---

## 🔹 Dataset

- **Dataset:** Telco Customer Churn Dataset  
- **Source:** Kaggle  
- **Records:** ~7,000 customers  
- **Target Variable:** `Churn` (Yes/No)  
- **Key Features:**
  - Tenure
  - Monthly Charges
  - Total Charges
  - Contract Type
  - Internet Service
  - Payment Method
  - Demographic information

---

## 🔹 Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Joblib  
- Streamlit  

---

## 🔹 Project Workflow

1. Data Loading  
2. Data Cleaning and Preprocessing  
3. Feature Engineering  
4. Categorical Encoding (One-Hot Encoding)  
5. Numerical Feature Scaling (StandardScaler)  
6. Model Training using Logistic Regression  
7. Model Evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC  
8. Model and Preprocessor Saving  
9. Real-time Prediction using Streamlit UI  

---

## 🔹 Model Performance

- **Accuracy:** ~75%  
- **ROC-AUC Score:** ~0.84  
- **Churn Recall:** ~76%  

These scores indicate strong performance on imbalanced churn data.

---

## 🔹 Streamlit Web Interface Features

- User-friendly input form
- Manual customer data entry
- Real-time churn probability prediction
- Risk-level classification:
  - ✅ Low Risk
  - ⚠️ Medium Risk
  - ❌ High Risk

---

## 🔹 How to Run the Project Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
