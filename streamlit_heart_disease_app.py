# ================================================================================
# STREAMLIT APPLICATION FOR HEART DISEASE PREDICTION
# Lazy-load artifacts + NumPy shim + BLAS thread caps (Cloud-safe)
# ================================================================================

# --- Keep the process stable on Streamlit Cloud / manylinux ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# --- NumPy 2.x pickle compat: alias numpy._core -> numpy.core (harmless on NumPy 1.x) ---
import sys
try:
    import numpy as _np  # noqa
    if "numpy._core" not in sys.modules:
        import numpy.core as _np_core  # noqa
        sys.modules["numpy._core"] = _np_core
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore")

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
    .main-header { font-size: 3rem; color: #1f77b4; text-align: center; margin-bottom: 2rem; font-weight: bold; }
    .sub-header  { font-size: 1.5rem; color: #ff7f0e; margin-bottom: 1rem; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; border-left: 5px solid #1f77b4; margin: 0.5rem 0; }
    .risk-high   { background-color: #ffebee;  border-left-color: #f44336; }
    .risk-medium { background-color: #fff3e0;  border-left-color: #ff9800; }
    .risk-low    { background-color: #e8f5e8;  border-left-color: #4caf50; }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# Artifact discovery (robust)
# ================================================================================

def _find_deploy_dir() -> Path:
    """
    Cari direktori artefak secara robust. Jika file ada di root repo, gunakan root.
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

    return base  # fallback ke root; error akan ditampilkan bila file tidak ada

DEPLOY_DIR = _find_deploy_dir()

# ================================================================================
# Lightweight: hanya baca deployment_info (tanpa load model) → cepat untuk health-check
# ================================================================================

def load_deployment_info_only() -> dict:
    info = {}
    for p in [
        DEPLOY_DIR / "deployment_info.json",
        DEPLOY_DIR / "deployment_info.pkl",
    ]:
        if p.exists():
            try:
                if p.suffix == ".json":
                    info = json.loads(p.read_text())
                else:
                    with open(p, "rb") as f:
                        info = pickle.load(f)
            except Exception as e:
                st.warning(f"Cannot read {p.name}: {e}")
            break
    return info

deployment_info = load_deployment_info_only()

# ================================================================================
# Heavy: load model & scaler (LAZY). Hanya dijalankan saat user klik Predict.
# ================================================================================

def _load_joblib_any(candidates, what):
    errs = []
    for p in candidates:
        try:
            if p.exists():
                return joblib.load(p)
        except ModuleNotFoundError as e:
            # handle numpy._core alias case
            if "numpy._core" in str(e):
                try:
                    import numpy.core as _np_core  # noqa
                    sys.modules["numpy._core"] = _np_core
                    return joblib.load(p)
                except Exception as e2:
                    errs.append(f"{p} -> {e2.__class__.__name__}: {e2}")
            else:
                errs.append(f"{p} -> {e.__class__.__name__}: {e}")
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
    """
    Load model, scaler, OPTIONAL bias_results.
    Catatan: function ini berat → dipanggil hanya saat diperlukan.
    """
    st.info(f"Using deployment dir: {DEPLOY_DIR}")

    model = _load_joblib_any([
        DEPLOY_DIR / "best_original_joblib.pkl",
        DEPLOY_DIR / "best_original_model_joblib.pkl",
        DEPLOY_DIR / "best_original.pkl",
    ], "best_original model")

    scaler = _load_joblib_any([
        DEPLOY_DIR / "scaler_joblib.pkl",
        DEPLOY_DIR / "scaler.pkl",
        DEPLOY_DIR / "standard_scaler.pkl",
    ], "scaler (StandardScaler)")

    # Optional bias results
    bias_results = None
    for bp in [
        DEPLOY_DIR / "bias_generalizability_results.pkl",
        DEPLOY_DIR.parent / "bias_generalizability_results.pkl",
        Path.cwd() / "heart_disease_analysis_results" / "bias_generalizability_results.pkl",
    ]:
        if bp.exists():
            try:
                with open(bp, "rb") as f:
                    bias_results = pickle.load(f)
            except Exception:
                pass
            break

    return model, scaler, bias_results

# ================================================================================
# Preprocessing & utilities
# ================================================================================

def preprocess_input(input_data: dict, selected_features: list, scaler):
    """
    Mirror training preprocessing:
    - Categorical → numeric (fixed OHE)
    - Engineered features
    - Pastikan urutan selected_features
    - Scale dengan scaler terlatih
    """
    df = pd.DataFrame([input_data]).copy()

    # Binary encodings
    if "Sex" in df.columns:
        df["Sex"] = 1 if str(df["Sex"].iloc[0]).upper() == "M" else 0
    if "ExerciseAngina" in df.columns:
        df["ExerciseAngina"] = 1 if str(df["ExerciseAngina"].iloc[0]).upper() == "Y" else 0

    def ensure_ohe(df_, col, categories):
        if col in df_.columns:
            val = str(df_[col].iloc[0])
            for cat in categories:
                df_[f"{col}_{cat}"] = 1 if val == cat else 0
            df_.drop(columns=[col], inplace=True)
        else:
            for cat in categories:
                df_[f"{col}_{cat}"] = 0

    ensure_ohe(df, "ChestPainType", ["ATA", "NAP", "ASY", "TA"])
    ensure_ohe(df, "RestingECG",    ["Normal", "ST", "LVH"])
    ensure_ohe(df, "ST_Slope",      ["Up", "Flat", "Down"])

    # Engineered features
    age  = float(df.get("Age", pd.Series([0])).iloc[0])
    mxhr = float(df.get("MaxHR", pd.Series([0])).iloc[0])
    rbp  = float(df.get("RestingBP", pd.Series([0])).iloc[0])
    chol = float(df.get("Cholesterol", pd.Series([0])).iloc[0])
    opk  = float(df.get("Oldpeak", pd.Series([0])).iloc[0])

    df["Fitness_Index"] = (mxhr / age) if age not in (0, 0.0) else 0.0
    df["Risk_Score"] = (age * 0.1) + (rbp * 0.01) + (chol * 0.001) + ((220.0 - mxhr) * 0.05) + (opk * 10.0)

    # Age groups
    df["Age_Group_Young"]         = 1 if 0 <= age < 40 else 0
    df["Age_Group_Middle_Young"]  = 1 if 40 <= age < 50 else 0
    df["Age_Group_Middle_Old"]    = 1 if 50 <= age < 60 else 0
    df["Age_Group_Old"]           = 1 if age >= 60 else 0

    # BP categories
    df["BP_Normal"]     = 1 if 0 <= rbp < 120 else 0
    df["BP_Elevated"]   = 1 if 120 <= rbp < 140 else 0
    df["BP_High"]       = 1 if 140 <= rbp < 180 else 0
    df["BP_Very_High"]  = 1 if 180 <= rbp else 0

    # Chol categories
    df["Chol_Desirable"]  = 1 if 0 <= chol < 200 else 0
    df["Chol_Borderline"] = 1 if 200 <= chol < 240 else 0
    df["Chol_High"]       = 1 if 240 <= chol else 0

    # Feature schema
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

    try:
        df_scaled = scaler.transform(df_selected)
    except Exception as e:
        st.error(f"Scaler transform failed: {e}")
        st.stop()

    return df_scaled, df


def calculate_risk_factors(input_data: dict):
    rf = []
    if input_data["Age"] > 65: rf.append("Advanced age (>65)")
    elif input_data["Age"] > 55: rf.append("Moderate age risk (55-65)")
    if input_data["Sex"] == "M": rf.append("Male gender")
    if input_data["RestingBP"] > 140: rf.append("High blood pressure (>140)")
    elif input_data["RestingBP"] > 120: rf.append("Elevated blood pressure (120-140)")
    if input_data["Cholesterol"] > 240: rf.append("High cholesterol (>240)")
    elif input_data["Cholesterol"] > 200: rf.append("Borderline cholesterol (200-240)")
    if input_data["FastingBS"] == 1: rf.append("Diabetes (Fasting BS >120)")
    if input_data["MaxHR"] < 100: rf.append("Low exercise capacity")
    if input_data["ExerciseAngina"] == "Y": rf.append("Exercise-induced angina")
    if input_data["Oldpeak"] > 2: rf.append("Significant ST depression")
    if input_data["ChestPainType"] == "ASY": rf.append("Asymptomatic chest pain")
    elif input_data["ChestPainType"] == "ATA": rf.append("Atypical angina")
    return rf


def interpret_prediction(probability: float):
    if probability >= 0.8:  return "Very High", "#d32f2f", "Immediate medical attention recommended. Consider cardiac evaluation and stress testing."
    if probability >= 0.6:  return "High", "#f57c00", "High risk detected. Consult cardiologist for comprehensive evaluation."
    if probability >= 0.4:  return "Moderate", "#fbc02d", "Moderate risk. Lifestyle modifications and regular monitoring advised."
    if probability >= 0.2:  return "Low-Moderate", "#689f38", "Low-moderate risk. Continue healthy lifestyle and routine check-ups."
    return "Low", "#388e3c", "Low risk. Maintain current healthy habits and regular preventive care."

# ================================================================================
# SIDEBAR – PATIENT INPUT (lightweight; tidak memicu load model)
# ================================================================================

st.sidebar.markdown("## 👤 Patient Information")
st.sidebar.markdown("### Basic Demographics")
age = st.sidebar.number_input("Age", min_value=20, max_value=100, value=50, step=1)
sex = st.sidebar.selectbox("Sex", ["M", "F"])
st.sidebar.markdown("### Symptoms & Clinical Presentation")
chest_pain = st.sidebar.selectbox("Chest Pain Type", ["ATA","NAP","ASY","TA"])
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["N","Y"])
st.sidebar.markdown("### Vital Signs & Laboratory")
resting_bp  = st.sidebar.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200, step=1)
fasting_bs  = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1])
st.sidebar.markdown("### Cardiac Test Results")
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal","ST","LVH"])
max_hr      = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
oldpeak     = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
st_slope    = st.sidebar.selectbox("ST Slope", ["Up","Flat","Down"])

col_btn1, col_btn2 = st.sidebar.columns(2)
predict_button = col_btn1.button("🔍 Analyze", type="primary")
preload_button  = col_btn2.button("⚡ Preload Models")

# ================================================================================
# MAIN CONTENT
# ================================================================================

st.markdown('<p class="main-header">❤️ Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown("### Advanced AI-Powered Cardiac Risk Assessment with Bias Monitoring")

with st.expander("ℹ️ About This System", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        algo = deployment_info.get("best_original_model", "—")
        perf = deployment_info.get("model_performance", {}).get("original", {})
        auc  = perf.get("roc_auc", np.nan)
        acc  = perf.get("accuracy", np.nan)
        n_tr = deployment_info.get("n_train", "—")
        st.markdown(f"**Algorithm:** {algo}<br>**Training Data:** {n_tr}<br>**AUC:** {auc:.3f} &nbsp;&nbsp; **Accuracy:** {acc:.3f}", unsafe_allow_html=True)
    with col2:
        st.markdown("- Comprehensive bias assessment\n- External validation tested\n- Real-time monitoring\n- Clinical decision support")

# Optional: preload to warm cache tanpa menunggu tombol Predict
if preload_button:
    with st.spinner("Loading models into cache..."):
        _ = load_models_and_data()
    st.success("Models preloaded.")

if predict_button:
    # Kumpulkan input
    input_data = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak,
        "ST_Slope": st_slope,
    }

    selected_features = deployment_info.get("selected_features", [])
    if not selected_features:
        st.error("`selected_features` tidak ditemukan di deployment_info.json/pkl.")
        st.stop()

    with st.spinner("Loading model & scaler..."):
        model, scaler, bias_results = load_models_and_data()

    try:
        processed_data, df_processed = preprocess_input(input_data, selected_features, scaler)
        prediction_proba  = model.predict_proba(processed_data)[0]
        prediction_binary = int(model.predict(processed_data)[0])

        risk_level, risk_color, recommendation = interpret_prediction(prediction_proba[1])
        risk_factors = calculate_risk_factors(input_data)

        st.markdown("---")
        st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns([2, 2, 3])
        with col1:
            css = "risk-high" if prediction_proba[1] >= 0.6 else ("risk-medium" if prediction_proba[1] >= 0.3 else "risk-low")
            st.markdown(f"""
            <div class="metric-card {css}">
                <h3 style="margin:0;color:{risk_color};">Risk Level: {risk_level}</h3>
                <h2 style="margin:10px 0;color:{risk_color};">{prediction_proba[1]:.1%}</h2>
                <p style="margin:0;">Probability of Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">Binary Classification</h3>
                <h2 style="margin:10px 0;color:{'#f44336' if prediction_binary==1 else '#4caf50'};">
                    {'Positive' if prediction_binary==1 else 'Negative'}
                </h2>
                <p style="margin:0;">{'Disease Detected' if prediction_binary==1 else 'No Disease Detected'}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin:0;">Clinical Recommendation</h3>
                <p style="margin:10px 0;font-size:14px;">{recommendation}</p>
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
            risk_df = pd.DataFrame({"Risk Factor": risk_factors, "Present": [1]*len(risk_factors)})
            fig_risk = px.bar(risk_df, y="Risk Factor", x="Present", orientation="h",
                              title="Patient Risk Factor Profile", color_discrete_sequence=["#ff7f0e"])
            fig_risk.update_layout(showlegend=False, xaxis_title="", height=max(300, len(risk_factors)*40))
            fig_risk.update_xaxis(showticklabels=False)
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("### 🔍 Feature Contribution (Scaled Values)")
        feat_names = selected_features[:10]
        feat_vals  = processed_data[0][:len(feat_names)]
        fig_features = go.Figure(go.Bar(x=feat_vals, y=feat_names, orientation="h", marker_color="lightblue"))
        fig_features.update_layout(title="Top Feature Values (Scaled)", xaxis_title="Scaled Value",
                                   yaxis_title="Features", height=420)
        st.plotly_chart(fig_features, use_container_width=True)

        st.markdown("### 📈 Model Performance & Reliability")
        perf = deployment_info.get("model_performance", {}).get("original", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model Accuracy", f"{perf.get('accuracy', np.nan):.3f}")
        c2.metric("AUC Score",     f"{perf.get('roc_auc', np.nan):.3f}")
        c3.metric("Precision",     f"{perf.get('precision', np.nan):.3f}")
        c4.metric("Recall",        f"{perf.get('recall', np.nan):.3f}")

        if bias_results is not None:
            st.markdown("### ⚖️ Fairness & Bias Monitoring")
            with st.expander("View Bias Assessment Results", expanded=False):
                st.markdown("""
                **Fairness Metrics Summary:**
                - Demographic parity across groups
                - Equalized odds evaluation
                - Accuracy parity across demographics
                """)

        # Summary & disclaimer
        st.markdown("### 📋 Patient Summary")
        sex_word = "male" if sex == "M" else "female"
        st.markdown(f"""
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
        """)
        st.warning("""
        **⚠️ Medical Disclaimer:** This prediction is for informational purposes only and should not replace professional medical advice.
        """)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.error("Please ensure model/scaler/feature schema are consistent with training.")


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
