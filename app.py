import streamlit as st
import pandas as pd
import joblib

# ------------------ LOAD MODEL ------------------
model = joblib.load("model/heart_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Heart Disease Predictor ❤️", layout="centered")

# ------------------ DARK FUTURISTIC UI FIX ------------------
st.markdown("""
    <style>
    /* Remove unwanted top padding & box */
    [data-testid="stHeader"] {
        display: none;
    }
    .block-container {
        padding-top: 1rem;
    }

    /* Background - dark gradient */
    body {
        background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
        color: #e0e0e0;
        font-family: 'Poppins', sans-serif;
    }

    /* Main card */
    .main {
        background: rgba(20, 20, 20, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2.5rem;
        box-shadow: 0 8px 25px rgba(0,0,0,0.6);
    }

    h1 {
        text-align: center;
        color: #00e6ff;
        font-size: 2.6rem !important;
        text-shadow: 0 0 10px rgba(0,230,255,0.6);
        margin-bottom: 0.3em;
    }

    h3 {
        text-align: center;
        color: #ccc;
        font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Inputs */
    .stSelectbox, .stNumberInput {
        color: #fff !important;
    }

    /* Button */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #00e6ff, #007bff);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.9em;
        font-size: 1.1em;
        font-weight: 600;
        box-shadow: 0 0 15px rgba(0, 123, 255, 0.4);
        transition: all 0.3s ease-in-out;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #007bff, #00e6ff);
        transform: scale(1.05);
        box-shadow: 0 0 25px rgba(0, 230, 255, 0.8);
    }

    /* Result cards */
    .result-card {
        margin-top: 1.5rem;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        color: white;
        font-size: 1.3rem;
        font-weight: 600;
        backdrop-filter: blur(15px);
        box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    }
    .high-risk {
        background: linear-gradient(135deg, #ff0844, #ffb199);
    }
    .low-risk {
        background: linear-gradient(135deg, #11998e, #38ef7d);
    }

    .footer {
        text-align: center;
        color: #bbb;
        font-size: 0.85rem;
        margin-top: 2rem;
        opacity: 0.9;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown("<h1>❤️ Heart Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter patient details below to assess heart disease risk</h3>", unsafe_allow_html=True)

# ------------------ INPUT FORM ------------------
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", 20, 100, 50)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["typical angina", "atypical angina", "non-anginal", "asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Cholesterol (mg/dl)", 100, 400, 240)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [True, False])

        with col2:
            restecg = st.selectbox("Resting ECG Result", ["normal", "lv hypertrophy", "ST-T abnormality"])
            thalch = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
            exang = st.selectbox("Exercise Induced Angina", [True, False])
            oldpeak = st.number_input("Oldpeak (ST Depression)", 0.0, 6.5, 1.0)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])
            ca = st.selectbox("Major Vessels (0–3)", [0, 1, 2, 3])
            thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🔍 Predict Now")

    st.markdown("</div>", unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if submitted:
    input_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "cp": {"typical angina": 0, "atypical angina": 1, "non-anginal": 2, "asymptomatic": 3}[cp],
        "trestbps": trestbps,
        "chol": chol,
        "fbs": 1 if fbs else 0,
        "restecg": {"normal": 0, "lv hypertrophy": 1, "ST-T abnormality": 2}[restecg],
        "thalch": thalch,
        "exang": 1 if exang else 0,
        "oldpeak": oldpeak,
        "slope": {"upsloping": 0, "flat": 1, "downsloping": 2}[slope],
        "ca": ca,
        "thal": {"normal": 0, "fixed defect": 1, "reversable defect": 2}[thal]
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[scaler.feature_names_in_]

    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    if prediction >= 1:
        st.markdown(
            f"<div class='result-card high-risk'>💔 High Risk of Heart Disease<br><br>Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='result-card low-risk'>💚 Low Risk of Heart Disease<br><br>Probability: {probability:.2f}</div>",
            unsafe_allow_html=True
        )

# ------------------ FOOTER ------------------
st.markdown("<p class='footer'>⚠️ This is a demonstration ML model — not for medical diagnosis.</p>", unsafe_allow_html=True)
