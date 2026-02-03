# 🫀 Heart Disease Risk Prediction System

An advanced web application built with **Streamlit** that predicts the risk of heart disease using a machine learning model (Random Forest) trained on the UCI Heart Disease dataset.

**Features**
- User-friendly sidebar inputs for all 13 clinical features
- Real-time risk prediction with probability and confidence visualization
- Color-coded risk levels (Low / Moderate / High / Very High)
- Comparison of entered values vs typical healthy references
- Clean tabs + modern UI styling
- Trained model, scaler, and feature selector saved as `.pkl` files

## Project Structure
Heart Disease Prediction System/
├── app.py                  # Streamlit web application
├── train_model.py          # Script to train & save the model
├── heart_model.pkl         # Trained RandomForest model
├── scaler.pkl              # Fitted StandardScaler
├── selector.pkl            # Fitted SelectKBest (feature selector)
├── requirements.txt        # Python dependencies
└── README.md               # This file

