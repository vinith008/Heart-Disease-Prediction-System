import streamlit as st
import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ───────────────────────────────────────────────
# Page Configuration & Custom Styling
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .big-font {font-size:24px !important; font-weight:bold; margin-bottom:10px;}
    .risk-very-high {color: #c92a2a; font-weight:bold; font-size:28px;}
    .risk-high      {color: #e03131; font-weight:bold; font-size:26px;}
    .risk-moderate  {color: #f59f00; font-weight:bold; font-size:24px;}
    .risk-low       {color: #37b24d; font-weight:bold; font-size:24px;}
    .stProgress > div > div > div {background-color: #e03131 !important;}
    .stProgress.low > div > div > div {background-color: #37b24d !important;}
    .card {background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
    </style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Load Model & Preprocessors (cached)
# ───────────────────────────────────────────────
@st.cache_resource
def load_model_and_preprocessors():
    try:
        model = pickle.load(open("heart_model.pkl", "rb"))
        scaler = pickle.load(open("scaler.pkl", "rb"))
        selector = pickle.load(open("selector.pkl", "rb"))
        return model, scaler, selector
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please run train_model.py first to generate heart_model.pkl, scaler.pkl, and selector.pkl")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, scaler, selector = load_model_and_preprocessors()

# ───────────────────────────────────────────────
# Sidebar - Patient Input
# ───────────────────────────────────────────────
st.sidebar.header("🩺 Patient Information")

with st.sidebar.container():
    st.markdown('<div class="big-font">Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 90, 55, 1)
    with col2:
        sex_choice = st.radio("Biological Sex", ["Female", "Male"], horizontal=True)
        sex = 1 if sex_choice == "Male" else 0

with st.sidebar.expander("Symptoms", expanded=True):
    cp = st.selectbox(
        "Chest Pain Type",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 - Typical Angina",
            1: "1 - Atypical Angina",
            2: "2 - Non-anginal Pain",
            3: "3 - Asymptomatic"
        }[x]
    )

with st.sidebar.expander("Measurements", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        trestbps = st.slider("Resting Blood Pressure (mm Hg)", 80, 220, 130, 5)
        chol = st.slider("Serum Cholesterol (mg/dl)", 100, 450, 240, 10)
    with col2:
        thalach = st.slider("Maximum Heart Rate Achieved", 60, 220, 150, 5)
        oldpeak = st.slider("ST Depression Induced", 0.0, 6.0, 1.0, 0.1)
    with col3:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "Yes" if x else "No")
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "Yes" if x else "No")

with st.sidebar.expander("ECG & Other Findings"):
    col1, col2 = st.columns(2)
    with col1:
        restecg = st.selectbox(
            "Resting ECG Results",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Normal", 1: "ST-T Wave Abnormality", 2: "Probable LVH"}[x]
        )
        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            options=[0, 1, 2],
            format_func=lambda x: {0: "Upsloping", 1: "Flat", 2: "Downsloping"}[x]
        )
    with col2:
        ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
        thal = st.selectbox(
            "Thalassemia",
            options=[1, 2, 3],
            format_func=lambda x: {1: "Normal", 2: "Fixed Defect", 3: "Reversible Defect"}[x]
        )

# ───────────────────────────────────────────────
# Prepare Input Data
# ───────────────────────────────────────────────
input_data = np.array([[
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]])

# Preprocessing pipeline (same as training)
selected_features = selector.transform(input_data)
scaled_features = scaler.transform(selected_features)

# ───────────────────────────────────────────────
# Main Content
# ───────────────────────────────────────────────
st.title("🫀 Advanced AI Heart Disease Risk Predictor")
st.caption("For educational and awareness purposes only — not a medical diagnosis tool")

if st.button("Calculate Risk Level", type="primary", use_container_width=True):
    with st.spinner("Analyzing patient data..."):
        # Prediction
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]
        heart_disease_prob = probabilities[1] * 100

        # Risk categorization
        if heart_disease_prob >= 85:
            level = "Very High Risk"
            class_name = "risk-very-high"
        elif heart_disease_prob >= 65:
            level = "High Risk"
            class_name = "risk-high"
        elif heart_disease_prob >= 40:
            level = "Moderate Risk"
            class_name = "risk-moderate"
        else:
            level = "Low Risk"
            class_name = "risk-low"

        # Display result
        st.markdown(f"<h2 class='{class_name}'>{level} ({heart_disease_prob:.1f}%)</h2>", unsafe_allow_html=True)

        # Progress bar
        progress_color_class = "low" if heart_disease_prob < 40 else ""
        st.markdown(f'<div class="stProgress {progress_color_class}">', unsafe_allow_html=True)
        st.progress(int(heart_disease_prob))
        st.markdown('</div>', unsafe_allow_html=True)

        # Probability bar chart
        st.subheader("Prediction Confidence")
        prob_df = pd.DataFrame({
            "Category": ["No Heart Disease", "Heart Disease"],
            "Probability (%)": [probabilities[0]*100, probabilities[1]*100]
        })
        st.bar_chart(prob_df.set_index("Category"))

# Show entered values
st.subheader("Your Entered Values")
feature_names = ["Age", "Sex", "Chest Pain Type", "Resting BP", "Cholesterol",
                 "Fasting BS >120", "Rest ECG", "Max HR", "Exer. Angina",
                 "Oldpeak", "Slope", "Major Vessels", "Thalassemia"]

display_values = [age, sex_choice, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]

df_display = pd.DataFrame({
    "Feature": feature_names,
    "Value": display_values
})

st.dataframe(df_display, hide_index=True, use_container_width=True)

# Disclaimer
st.markdown("---")
st.info("""
**Important Medical Disclaimer**  
This application is for **educational and awareness purposes only**.  
It is **not** a substitute for professional medical advice, diagnosis, or treatment.  
Always consult a qualified cardiologist or healthcare provider.
""")
