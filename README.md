# 🩺 Diabetes Risk Prediction 

This app predicts the likelihood of diabetes based on clinical input. It uses an XGBoost classifier trained on standardized numeric and one-hot encoded categorical data. The app includes warnings for out-of-distribution inputs based on EDA thresholds.

##  Features
- Machine learning model (XGBoost) for prediction
- StandardScaler for normalized numeric input
- Input validation based on EDA thresholds
- Deployment-ready Streamlit interface

##  Inputs
- Blood Glucose Level (11.5 - 247.5)
- HbA1c Level (2.7 - 8.3)
- BMI (14.7 - 38.5)
- Age
- Gender, Smoking History
- Hypertension, Heart Disease

##  Files Included

```
├── app.py                     # Streamlit app
├── diabetes_xgb_model.joblib  # Trained model
├── scaler.joblib              # Trained scaler
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

## ✍️ Author

**Noel Franck Kouadio**  
*Data Scientist 
