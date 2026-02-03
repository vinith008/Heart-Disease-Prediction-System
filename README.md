# Heart Disease Risk Prediction System

## Built With Streamlit

This web application is built using **Streamlit** to provide an interactive and user-friendly medical risk assessment interface.  
Users can enter patient details and instantly receive a heart disease risk prediction powered by a Machine Learning model.

---

## Features

- Interactive medical input form  
- Real-time heart disease risk prediction  
- Risk level classification (Low, Moderate, High, Very High)  
- Probability visualization with progress bar and charts  
- Clean and professional medical-style interface  
- Model confidence display  
- Educational medical disclaimer  
- Feature preprocessing using StandardScaler and SelectKBest  

---

## Tech Stack

**Frontend:** Streamlit  
**Backend / ML:** Python, scikit-learn  
**Model:** RandomForestClassifier  
**Data Processing:** pandas, numpy  
**Visualization:** matplotlib, seaborn  

---

## Project Structure

```

heart-disease-prediction/
│
├── app.py              # Streamlit web application
├── train_model.py      # Script to train and save the ML model
├── heart_model.pkl     # Trained model file (generated after training)
├── scaler.pkl          # Feature scaler (generated after training)
├── selector.pkl        # Feature selector (generated after training)
├── requirements.txt
└── README.md

````

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/heart-disease-prediction.git
cd heart-disease-prediction
````

### 2. Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

### Step 1: Train the Model

```bash
python train_model.py
```

This generates:

* `heart_model.pkl`
* `scaler.pkl`
* `selector.pkl`

### Step 2: Launch Streamlit App

```bash
streamlit run app.py
```

Open your browser and go to:

```
http://localhost:8501
```

---

## How to Use

1. Enter patient information in the sidebar
2. Click **"Calculate Risk Level"**
3. View the predicted heart disease risk and probability

---

## Model Information

The model is trained using a Heart Disease dataset and uses:

* Feature scaling for improved performance
* Feature selection to remove noise
* Random Forest ensemble learning for higher accuracy

---

## Important Disclaimer

This application is for **educational and demonstration purposes only**.

It is **not** a medical device and must **not** be used for real medical diagnosis or treatment decisions.

Always consult a qualified healthcare professional for medical advice.

---

## Future Improvements

* Add Explainable AI (SHAP values)
* Downloadable medical report
* Model comparison dashboard
* Cloud deployment
* User authentication

---

## License

This project is open-source and available under the MIT License.

```

---

If you want, next I can give you a **perfect `requirements.txt`** file for this project.
```
