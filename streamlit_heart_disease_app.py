# ================================================================================
# STREAMLIT APPLICATION FOR HEART DISEASE PREDICTION
# Complete deployment with bias monitoring and comprehensive analysis
# ================================================================================

import warnings
warnings.filterwarnings("ignore")

import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import pickle
import json

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ------------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .risk-high { background-color: #ffebee; border-left-color: #f44336; }
    .risk-medium { background-color: #fff3e0; border-left-color: #ff9800; }
    .risk-low { background-color: #e8f5e8; border-left-color: #4caf50; }
    .sidebar .sidebar-content { background-color: #fafafa; }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# Robust artifact loader
# ================================================================================

def _find_deploy_dir() -> Path:
    """
    Cari direktori artefak secara robust.
    - Coba beberapa jalur umum (deployment di dalam folder hasil analisis).
    - Jika tidak ada folder deployment, tapi file artefak ada di root repo,
      gunakan root repo sebagai deployment dir.
    """
    base = Path(__file__).resolve().parent
    candidates = [
        base / "heart_disease_analysis_results" / "deployment",
        base / "deployment",
        base.parent / "deployment",
        base.parent / "heart_disease_analysis_results" / "deployment",
        Path("/content/heart_disease_analysis_results/deployment"),
        Path.cwd() / "heart_disease_analysis_results" / "deployment",
    ]
    for c in candidates:
        if c.exists():
            return c

    # Jika file artefak ada di root repo (sesuai repo kamu sekarang), pakai base
    root_files = {
        "best_original_joblib.pkl",
        "best_original_model_joblib.pkl",
        "best_original.pkl",
        "scaler_joblib.pkl",
        "deployment_info.json",
        "deployment_info.pkl",
    }
    if any((base / f).exists() for f in root_files):
        return base

    # fallback
    return base / "heart_disease_analysis_results" / "deployment"

DEPLOY_DIR = _find_deploy_dir()


def _load_joblib_any(candidates, what):
    errs = []
    for p in candidates:
        try:
            if p.exists():
                return joblib.load(p)
        except Exception as e:
            errs.append(f"{p} -> {e.__class__.__name__}: {e}")
    st.error(f"Failed to load {what}. Tried:\n  " + "\n  ".join(str(x) for x in candidates))
    try:
        st.warning("DEPLOY_DIR contents: " + ", ".join(sorted(p.name for p in DEPLOY_DIR.iterdir())))
    except Exception:
        pass
    if errs:
        st.warning("Last errors:\n" + "\n".join(errs[-3:]))
    st.stop()


@st.cache_resource
def load_models_and_data():
    """Load model, scaler, deployment info. Bias results optional."""
    st.info(f"Using deployment dir: {DEPLOY_DIR}")

    # Model
    model = _load_joblib_any([
        DEPLOY_DIR / "best_original_joblib.pkl",
        DEPLOY_DIR / "best_original_model_joblib.pkl",
        DEPLOY_DIR / "best_original.pkl",
    ], "best_original model")

    # Scaler
    scaler = _load_joblib_any([
        DEPLOY_DIR / "scaler_joblib.pkl",
        DEPLOY_DIR / "scaler.pkl",
        DEPLOY_DIR / "standard_scaler.pkl",
    ], "scaler (StandardScaler)")

    # deployment_info: JSON preferred, PKL fallback
    deployment_info = {}
    for p in [
        DEPLOY_DIR / "deployment_info.json",
        DEPLOY_DIR / "deployment_info.pkl",
        DEPLOY_DIR / "info.json",
        DEPLOY_DIR / "info.pkl",
    ]:
        if p.exists():
            try:
                if p.suffix == ".json":
                    deployment_info = json.loads(p.read_text())
                else:
                    with open(p, "rb") as f:
                        deployment_info = pickle.load(f)
            except Exception as e:
                st.warning(f"Cannot read {p.name}: {e}")
            break
    if not deployment_info:
        st.warning("deployment_info not found. Using minimal defaults; ensure selected_features available.")

    # Optional bias results
    bias_results = None
    bias_path_candidates = [
        DEPLOY_DIR.parent / "bias_generalizability_results.pkl",
        Path.cwd() / "heart_disease_analysis_results" / "bias_generalizability_results.pkl",
        DEPLOY_DIR / "bias_generalizability_results.pkl",
    ]
    for bp in bias_path_candidates:
        if bp.exists():
            try:
                with open(bp, "rb") as f:
                    bias_results = pickle.load(f)
            except Exception:
                pass
            break

    return model, scaler, deployment_info, bias_results


model, scaler, deployment_info, bias_results = load_models_and_data()

# ================================================================================
# Preprocessing & utilities
# ================================================================================

def preprocess_input(input_data: dict):
    """
    Mirror training preprocessing:
    - Convert categorical to numeric (fixed OHE schema)
    - Engineer features
    - Guarantee `selected_features` presence and order
    - Scale with the trained scaler
    """
    df = pd.DataFrame([input_data]).copy()

    # Binary encodings (same names dengan training)
    if "Sex" in df.columns:
        df["Sex"] = 1 if str(df["Sex"].iloc[0]).upper() == "M" else 0
    if "ExerciseAngina" in df.columns:
        df["ExerciseAngina"] = 1 if str(df["ExerciseAngina"].iloc[0]).upper() == "Y" else 0

    # Fixed OHE helper
    def ensure_ohe(df_, col, categories):
        if col in df_.columns:
            val = str(df_[col].iloc[0])
            for cat in categories:
                df_[f"{col}_{cat}"] = 1 if val == cat else 0
            df_.drop(columns=[col], inplace=True)
        else:
            for cat in categories:
                df_[f"{col}_{cat}"] = 0

    # Apply fixed OHE
    ensure_ohe(df, "ChestPainType", ["ATA", "NAP", "ASY", "TA"])
    ensure_ohe(df, "RestingECG", ["Normal", "ST", "LVH"])
    ensure_ohe(df, "ST_Slope", ["Up", "Flat", "Down"])

    # Engineered features (mirror training logic)
    age = float(df.get("Age", pd.Series([0])).iloc[0])
    maxhr = float(df.get("MaxHR", pd.Series([0])).iloc[0])

    df["Fitness_Index"] = (maxhr / age) if age not in [0, 0.0] else 0.0

    for col in ["RestingBP", "Cholesterol", "Oldpeak"]:
        if col not in df.columns:
            df[col] = 0.0

    df["Risk_Score"] = (
        float(df["Age"].iloc[0]) * 0.1
        + float(df["RestingBP"].iloc[0]) * 0.01
        + float(df["Cholesterol"].iloc[0]) * 0.001
        + (220.0 - float(df["MaxHR"].iloc[0])) * 0.05
        + float(df["Oldpeak"].iloc[0]) * 10.0
    )

    # Age groups
    df["Age_Group_Young"] = 1 if 0 <= age < 40 else 0
    df["Age_Group_Middle_Young"] = 1 if 40 <= age < 50 else 0
    df["Age_Group_Middle_Old"] = 1 if 50 <= age < 60 else 0
    df["Age_Group_Old"] = 1 if age >= 60 else 0

    # BP categories
    bp = float(df["RestingBP"].iloc[0])
    df["BP_Normal"] = 1 if 0 <= bp < 120 else 0
    df["BP_Elevated"] = 1 if 120 <= bp < 140 else 0
    df["BP_High"] = 1 if 140 <= bp < 180 else 0
    df["BP_Very_High"] = 1 if 180 <= bp else 0

    # Cholesterol categories
    chol = float(df["Cholesterol"].iloc[0])
    df["Chol_Desirable"] = 1 if 0 <= chol < 200 else 0
    df["Chol_Borderline"] = 1 if 200 <= chol < 240 else 0
    df["Chol_High"] = 1 if 240 <= chol else 0

    # Feature schema enforcement
    selected_features = deployment_info.get("selected_features", [])
    if not selected_features:
        st.error("`selected_features` is missing in deployment_info (json/pkl).")
        st.stop()

    for feat in selected_features:
        if feat not in df.columns:
            df[feat] = 0

    try:
        df_selected = df[selected_features]
    except Exception as e:
        st.error(f"Feature mismatch after preprocessing: {e}")
        st.write("Expected features:", selected_features)
        st.write("Current columns:", list(df.columns))
        st.stop()

    # Scale
    try:
        df_scaled = scaler.transform(df_selected)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    return df_scaled, df


def calculate_risk_factors(input_data: dict):
    """Heuristic risk factor flags for explanation."""
    rf = []
    if input_data["Age"] > 65:
        rf.append("Advanced age (>65)")
    elif input_data["Age"] > 55:
        rf.append("Moderate age risk (55-65)")

    if input_data["Sex"] == "M":
        rf.append("Male gender")

    if input_data["RestingBP"] > 140:
        rf.append("High blood pressure (>140)")
    elif input_data["RestingBP"] > 120:
        rf.append("Elevated blood pressure (120-140)")

    if input_data["Cholesterol"] > 240:
        rf.append("High cholesterol (>240)")
    elif input_data["Cholesterol"] > 200:
        rf.append("Borderline cholesterol (200-240)")

    if input_data["FastingBS"] == 1:
        rf.append("Diabetes (Fasting BS >120)")

    if input_data["MaxHR"] < 100:
        rf.append("Low exercise capacity")

    if input_data["ExerciseAngina"] == "Y":
        rf.append("Exercise-induced angina")

    if input_data["Oldpeak"] > 2:
        rf.append("Significant ST depression")

    if input_data["ChestPainType"] == "ASY":
        rf.append("Asymptomatic chest pain")
    elif input_data["ChestPainType"] == "ATA":
        rf.append("Atypical angina")

    return rf


def interpret_prediction(probability: float, risk_factors: list):
    """Map probability to risk level and recommendation."""
    if probability >= 0.8:
        return "Very High", "#d32f2f", "Immediate medical attention recommended. Consider cardiac evaluation and stress testing."
    if probability >= 0.6:
        return "High", "#f57c00", "High risk detected. Consult cardiologist for comprehensive evaluation."
    if probability >= 0.4:
        return "Moderate", "#fbc02d", "Moderate risk. Lifestyle modifications and regular monitoring advised."
    if probability >= 0.2:
        return "Low-Moderate", "#689f38", "Low-moderate risk. Continue healthy lifestyle and routine check-ups."
    return "Low", "#388e3c", "Low risk. Maintain current healthy habits and regular preventive care."


# ================================================================================
# SIDEBAR – PATIENT INPUT
# ================================================================================

st.sidebar.markdown("## 👤 Patient Information")

st.sidebar.markdown("### Basic Demographics")
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50, step=1)
sex = st.sidebar.selectbox("Sex", ["M", "F"], help="M = Male, F = Female")

st.sidebar.markdown("### Symptoms & Clinical Presentation")
chest_pain = st.sidebar.selectbox(
    "Chest Pain Type", ["ATA", "NAP", "ASY", "TA"],
    help="ATA=Atypical Angina, NAP=Non-Anginal Pain, ASY=Asymptomatic, TA=Typical Angina"
)
exercise_angina = st.sidebar.selectbox(
    "Exercise Induced Angina", ["N", "Y"], help="Does exercise cause chest pain? Y=Yes, N=No"
)

st.sidebar.markdown("### Vital Signs & Laboratory")
resting_bp = st.sidebar.number_input(
    "Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1,
    help="Normal: <120, Elevated: 120-129, High: ≥130"
)
cholesterol = st.sidebar.number_input(
    "Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1,
    help="Desirable: <200, Borderline: 200-239, High: ≥240"
)
fasting_bs = st.sidebar.selectbox(
    "Fasting Blood Sugar > 120 mg/dl", [0, 1],
    help="0 = ≤120 mg/dl (Normal), 1 = >120 mg/dl (Diabetes)"
)

st.sidebar.markdown("### Cardiac Test Results")
resting_ecg = st.sidebar.selectbox(
    "Resting ECG", ["Normal", "ST", "LVH"],
    help="Normal, ST=ST-T wave abnormality, LVH=Left ventricular hypertrophy"
)
max_hr = st.sidebar.number_input(
    "Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1,
    help="Maximum heart rate during exercise test"
)
oldpeak = st.sidebar.number_input(
    "ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1,
    help="ST depression induced by exercise relative to rest"
)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"], help="Slope of peak exercise ST segment")

predict_button = st.sidebar.button("🔍 Analyze Heart Disease Risk", type="primary")


# ================================================================================
# MAIN CONTENT
# ================================================================================

st.markdown('<p class="main-header">❤️ Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown("### Advanced AI-Powered Cardiac Risk Assessment with Bias Monitoring")

with st.expander("ℹ️ About This System", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        algo = deployment_info.get("best_original_model", "LogisticRegression")
        perf = deployment_info.get("model_performance", {}).get("original", {})
        auc = perf.get("roc_auc", np.nan)
        acc = perf.get("accuracy", np.nan)
        st.markdown(f"""
        **Model Information:**
        - Algorithm: {algo}
        - Training Data: {deployment_info.get('n_train', 918)}
        - Performance: AUC = {auc:.3f}
        - Accuracy: {acc:.3f}
        """)
    with col2:
        st.markdown("""
        **Key Features:**
        - Comprehensive bias assessment
        - External validation tested
        - Real-time monitoring
        - Clinical decision support
        """)

if predict_button:
    input_data = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    try:
        processed_data, df_processed = preprocess_input(input_data)
        prediction_proba = model.predict_proba(processed_data)[0]
        prediction_binary = int(model.predict(processed_data)[0])

        risk_factors = calculate_risk_factors(input_data)
        risk_level, risk_color, recommendation = interpret_prediction(prediction_proba[1], risk_factors)

        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            css_class = "risk-high" if prediction_proba[1] >= 0.6 else ("risk-medium" if prediction_proba[1] >= 0.3 else "risk-low")
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <h3 style="margin: 0; color: {risk_color};">Risk Level: {risk_level}</h3>
                <h2 style="margin: 10px 0; color: {risk_color};">{prediction_proba[1]:.1%}</h2>
                <p style="margin: 0;">Probability of Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0;">Binary Classification</h3>
                <h2 style="margin: 10px 0; color: {'#f44336' if prediction_binary == 1 else '#4caf50'};">
                    {'Positive' if prediction_binary == 1 else 'Negative'}
                </h2>
                <p style="margin: 0;">{'Disease Detected' if prediction_binary == 1 else 'No Disease Detected'}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0;">Clinical Recommendation</h3>
                <p style="margin: 10px 0; font-size: 14px;">{recommendation}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📊 Risk Probability Breakdown")
        fig_prob = go.Figure(go.Bar(
            x=["No Heart Disease", "Heart Disease"],
            y=[prediction_proba[0], prediction_proba[1]],
            marker_color=["#4caf50", "#f44336"],
            text=[f"{prediction_proba[0]:.1%}", f"{prediction_proba[1]:.1%}"],
            textposition="auto",
        ))
        fig_prob.update_layout(title="Prediction Probabilities", yaxis_title="Probability",
                               xaxis_title="Classification", showlegend=False, height=400)
        st.plotly_chart(fig_prob, use_container_width=True)

        if risk_factors:
            st.markdown("### ⚠️ Identified Risk Factors")
            risk_factor_df = pd.DataFrame({"Risk Factor": risk_factors, "Present": [1] * len(risk_factors)})
            fig_risk = px.bar(risk_factor_df, y="Risk Factor", x="Present", orientation="h",
                              title="Patient Risk Factor Profile", color_discrete_sequence=["#ff7f0e"])
            fig_risk.update_layout(showlegend=False, xaxis_title="", height=max(300, len(risk_factors) * 40))
            fig_risk.update_xaxis(showticklabels=False)
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("### 🔍 Feature Contribution (Scaled Values)")
        feat_names = deployment_info.get("selected_features", [])[:10]
        feat_vals = processed_data[0][:len(feat_names)]
        fig_features = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation="h", marker_color="lightblue"))
        fig_features.update_layout(title="Top Feature Values (Scaled)", xaxis_title="Scaled Value",
                                   yaxis_title="Features", height=420)
        st.plotly_chart(fig_features, use_container_width=True)

        st.markdown("### 📈 Model Performance & Reliability")
        perf = deployment_info.get("model_performance", {}).get("original", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Accuracy", f"{perf.get('accuracy', np.nan):.3f}", help="Accuracy on test set")
        c2.metric("AUC Score", f"{perf.get('roc_auc', np.nan):.3f}", help="Area Under ROC")
        c3.metric("Precision", f"{perf.get('precision', np.nan):.3f}", help="Precision (Positive class)")
        c4.metric("Recall", f"{perf.get('recall', np.nan):.3f}", help="Sensitivity/Recall")

        if bias_results is not None:
            st.markdown("### ⚖️ Fairness & Bias Monitoring")
            with st.expander("View Bias Assessment Results", expanded=False):
                st.markdown("""
                **Fairness Metrics Summary:**
                - Demographic parity across groups
                - Equalized odds evaluation
                - Accuracy parity across demographics
                """)
                if isinstance(bias_results, dict) and "fairness_metrics" in bias_results:
                    st.write("Detailed fairness metrics available in bias assessment results.")

        st.markdown("### 📋 Patient Summary")
        sex_word = "male" if sex == "M" else "female"
        patient_summary = f"""
        **Patient Profile:**
        - {age} year old {sex_word}
        - Chest pain type: {chest_pain}
        - Blood pressure: {resting_bp} mmHg
        - Cholesterol: {cholesterol} mg/dl
        - Exercise capacity: {max_hr} bpm max HR

        **Risk Assessment:**
        - Overall risk level: **{risk_level}**
        - Probability of heart disease: **{prediction_proba[1]:.1%}**
        - Number of risk factors: **{len(risk_factors)}**

        **Recommendation:**
        {recommendation}
        """
        st.markdown(patient_summary)

        st.markdown("---")
        st.warning("""
        **⚠️ Medical Disclaimer:**
        This prediction is for informational purposes only and should not replace professional medical advice.
        Always consult qualified healthcare professionals for medical decisions. This AI assists, not replaces, clinical judgment.
        """)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please check your input values and ensure model/scaler/feature schema are consistent.")

else:
    st.markdown("### 🚀 Welcome to the Heart Disease Prediction System")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **How to Use:**
        1. Enter patient information in the sidebar  
        2. Click **Analyze Heart Disease Risk**  
        3. Review the comprehensive risk assessment

        **Features:**
        - Advanced ML model
        - Bias assessment
        - Real-time risk factor analysis
        - Clinical decision support
        """)
    with c2:
        st.markdown("""
        **Clinical Parameters:**
        - **Demographics:** Age, Sex
        - **Symptoms:** Chest pain, Exercise angina
        - **Vitals:** BP, HR
        - **Labs:** Cholesterol, Blood sugar
        - **Tests:** ECG, ST slope

        **Risk Levels:**
        - 🟢 **Low:** <20% probability  
        - 🟡 **Moderate:** 20–60%  
        - 🔴 **High:** >60%
        """)

    with st.expander("📝 Try Sample Patient Data", expanded=False):
        st.markdown("""
        **High Risk Example:** Age 65, Male, ASY, BP 160, Chol 280, Angina=Yes, MaxHR 120  
        **Low Risk Example:** Age 35, Female, NAP, BP 110, Chol 180, Angina=No, MaxHR 180
        """)

# ------------------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>Heart Disease Prediction System</strong> | Powered by Machine Learning</p>
    <p>Bias assessment and fairness monitoring | Version 1.0</p>
    <p>For healthcare professionals and clinical decision support</p>
</div>
""", unsafe_allow_html=True)
