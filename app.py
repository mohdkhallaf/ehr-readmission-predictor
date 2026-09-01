import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from src.train_model import train_readmission_model

st.set_page_config(page_title="EHR Readmission Predictor", layout="wide")
st.title("🩺 EHR 30-Day Patient Readmission Predictor")
st.markdown("Predict readmission risks and inspect SHAP feature attribution for clinical decision support.")

@st.cache_resource
def load_engine():
    return train_readmission_model()

model, explainer, shap_values, X_test = load_engine()

st.sidebar.header("Patient Clinical Profile")
age = st.sidebar.slider("Age", 18, 95, 65)
los = st.sidebar.slider("Length of Stay (Days)", 1, 30, 6)
labs = st.sidebar.slider("Lab Procedures", 1, 100, 45)
meds = st.sidebar.slider("Number of Medications", 1, 40, 12)
er_visits = st.sidebar.slider("Prior ER Visits (Past Year)", 0, 10, 2)
diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])

patient_data = pd.DataFrame([{
    'age': age, 'length_of_stay_days': los, 'num_lab_procedures': labs,
    'num_medications': meds, 'num_emergency_visits_prior_year': er_visits,
    'has_diabetes': diabetes, 'has_hypertension': hypertension
}])

risk_prob = model.predict_proba(patient_data)[0][1]

col1, col2 = st.columns(2)
with col1:
    st.metric("30-Day Readmission Risk", f"{risk_prob * 100:.1f}%")
    if risk_prob > 0.5:
        st.error("⚠️ High Readmission Risk Flagged")
    else:
        st.success("✅ Low Readmission Risk")

with col2:
    st.subheader("Model Explainability (SHAP Values)")
    p_shap = explainer(patient_data)
    fig, ax = plt.subplots()
    shap.plots.waterfall(p_shap[0], show=False)
    st.pyplot(fig)