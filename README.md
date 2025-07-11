# 📉 Customer Churn Prediction App

This is a machine learning-powered web app built using Streamlit that predicts whether a telecom customer is likely to churn based on their service and demographic details.

## 🚀 Features

- Interactive web UI using Streamlit
- Predict customer churn using a trained Random Forest model
- Real-time input form for user-friendly predictions
- Model trained on the Telco Customer Churn dataset
- Probability-based churn predictions
- Deployed online using Streamlit Cloud

## 📁 Files Included

- `app.py` – Main Streamlit app file
- `best_model.pkl` – Trained Random Forest model
- `scaler.pkl` – Preprocessing scaler used during training
- `requirements.txt` – Python dependencies

## 📦 Installation (Local)

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction-app.git
cd customer-churn-prediction-app
pip install -r requirements.txt
streamlit run app.py
