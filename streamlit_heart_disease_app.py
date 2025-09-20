# streamlit_heart_disease_app.py
# ================================================================================
# HEART DISEASE PREDICTION SYSTEM - COMPLETE STREAMLIT APPLICATION
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff  # (opsional, tidak wajib terpakai)
from datetime import datetime, timedelta
import time
import json
import math

# ================================================================================
# PAGE CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Heart Disease Prediction System",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/heart-disease-prediction',
        'Report a bug': "https://github.com/your-repo/heart-disease-prediction/issues",
        'About': "# Heart Disease Prediction System\nAdvanced AI-powered clinical decision support tool."
    }
)

# ================================================================================
# CUSTOM CSS STYLING
# ================================================================================

st.markdown("""
<style>
    /* Main styling */
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #ff7f0e;
        margin-bottom: 1.5rem;
        font-weight: bold;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 10px;
    }
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    /* Risk level cards */
    .risk-very-high { background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); border-left-color: #d32f2f; }
    .risk-high      { background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%); border-left-color: #f57c00; }
    .risk-moderate  { background: linear-gradient(135deg, #fffde7 0%, #fff176 100%); border-left-color: #fbc02d; }
    .risk-low       { background: linear-gradient(135deg, #e8f5e8 0%, #a5d6a7 100%); border-left-color: #388e3c; }
    /* Sidebar styling */
    .sidebar .sidebar-content { background: linear-gradient(180deg, #fafafa 0%, #f0f2f6 100%); }
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #2e8bc0 100%);
        color: white; border: none; padding: 0.75rem 2rem;
        border-radius: 25px; font-weight: bold; transition: all 0.3s;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
    /* Info boxes */
    .info-box { background-color: #e3f2fd; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196f3; margin: 1rem 0; }
    .warning-box { background-color: #fff3e0; padding: 1rem; border-radius: 10px; border-left: 4px solid #ff9800; margin: 1rem 0; }
    .success-box { background-color: #e8f5e8; padding: 1rem; border-radius: 10px; border-left: 4px solid #4caf50; margin: 1rem 0; }
    /* Animation */
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .fade-in { animation: fadeIn 0.6s ease-out; }
    /* Progress indicators */
    .progress-text { text-align: center; font-weight: bold; color: #1f77b4; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# ADVANCED HEART DISEASE PREDICTION MODEL (RULES-BASED, TANPA FILE MODEL)
# ================================================================================

class AdvancedHeartDiseasePredictor:
    """
    Advanced clinical prediction model for heart disease risk assessment (rules-based).
    Stabil dan cocok untuk deployment cloud tanpa file model eksternal.
    """
    def __init__(self):
        # Weights (total ~1.38) — digunakan untuk skor lalu diproyeksikan via sigmoid
        self.feature_weights = {
            'age_factor': 0.16,
            'gender_factor': 0.13,
            'chest_pain_factor': 0.19,
            'blood_pressure_factor': 0.11,
            'cholesterol_factor': 0.12,
            'diabetes_factor': 0.08,
            'ecg_factor': 0.07,
            'exercise_capacity_factor': 0.14,
            'exercise_angina_factor': 0.15,
            'st_depression_factor': 0.13,
            'st_slope_factor': 0.10
        }
        self.thresholds = {
            'age_high_risk': 65,
            'age_moderate_risk': 55,
            'bp_high': 140,
            'bp_elevated': 120,
            'cholesterol_high': 240,
            'cholesterol_borderline': 200,
            'max_hr_low': 100,
            'oldpeak_significant': 2.0
        }

    def calculate_age_factor(self, age):
        if age >= self.thresholds['age_high_risk']:
            return 1.0
        elif age >= self.thresholds['age_moderate_risk']:
            return 0.6 + (age - self.thresholds['age_moderate_risk']) / 10 * 0.4
        else:
            return max(0.1, (age - 30) / 25 * 0.6)

    def calculate_gender_factor(self, sex):
        return 1.0 if sex == 'M' else 0.3

    def calculate_chest_pain_factor(self, chest_pain_type):
        pain_risk = {'ASY': 1.0, 'NAP': 0.4, 'ATA': 0.7, 'TA': 0.5}
        return pain_risk.get(chest_pain_type, 0.5)

    def calculate_bp_factor(self, resting_bp):
        if resting_bp >= 180: return 1.0
        if resting_bp >= self.thresholds['bp_high']: return 0.8
        if resting_bp >= self.thresholds['bp_elevated']: return 0.4
        return 0.1

    def calculate_cholesterol_factor(self, cholesterol):
        if cholesterol >= 300: return 1.0
        if cholesterol >= self.thresholds['cholesterol_high']: return 0.8
        if cholesterol >= self.thresholds['cholesterol_borderline']: return 0.5
        return 0.2

    def calculate_diabetes_factor(self, fasting_bs):
        return 0.8 if fasting_bs == 1 else 0.1

    def calculate_ecg_factor(self, resting_ecg):
        ecg_risk = {'Normal': 0.1, 'ST': 0.7, 'LVH': 0.9}
        return ecg_risk.get(resting_ecg, 0.1)

    def calculate_exercise_capacity_factor(self, max_hr, age):
        predicted_max_hr = max(1.0, 220 - age)  # guard div by zero
        hr_percentage = max_hr / predicted_max_hr
        if hr_percentage < 0.6: return 1.0
        if hr_percentage < 0.75: return 0.7
        if hr_percentage < 0.85: return 0.4
        return 0.1

    def calculate_exercise_angina_factor(self, exercise_angina):
        return 0.9 if exercise_angina == 'Y' else 0.1

    def calculate_st_depression_factor(self, oldpeak):
        if oldpeak >= 3.0: return 1.0
        if oldpeak >= 2.0: return 0.8
        if oldpeak >= 1.0: return 0.5
        return 0.1

    def calculate_st_slope_factor(self, st_slope):
        slope_risk = {'Down': 1.0, 'Flat': 0.7, 'Up': 0.2}
        return slope_risk.get(st_slope, 0.5)

    def predict_risk_probability(self, patient_data):
        # Individual risks
        age_risk = self.calculate_age_factor(patient_data['Age'])
        gender_risk = self.calculate_gender_factor(patient_data['Sex'])
        chest_pain_risk = self.calculate_chest_pain_factor(patient_data['ChestPainType'])
        bp_risk = self.calculate_bp_factor(patient_data['RestingBP'])
        chol_risk = self.calculate_cholesterol_factor(patient_data['Cholesterol'])
        diabetes_risk = self.calculate_diabetes_factor(patient_data['FastingBS'])
        ecg_risk = self.calculate_ecg_factor(patient_data['RestingECG'])
        exercise_capacity_risk = self.calculate_exercise_capacity_factor(patient_data['MaxHR'], patient_data['Age'])
        exercise_angina_risk = self.calculate_exercise_angina_factor(patient_data['ExerciseAngina'])
        st_depression_risk = self.calculate_st_depression_factor(patient_data['Oldpeak'])
        st_slope_risk = self.calculate_st_slope_factor(patient_data['ST_Slope'])

        # Weighted risk score
        w = self.feature_weights
        risk_score = (
            age_risk * w['age_factor'] +
            gender_risk * w['gender_factor'] +
            chest_pain_risk * w['chest_pain_factor'] +
            bp_risk * w['blood_pressure_factor'] +
            chol_risk * w['cholesterol_factor'] +
            diabetes_risk * w['diabetes_factor'] +
            ecg_risk * w['ecg_factor'] +
            exercise_capacity_risk * w['exercise_capacity_factor'] +
            exercise_angina_risk * w['exercise_angina_factor'] +
            st_depression_risk * w['st_depression_factor'] +
            st_slope_risk * w['st_slope_factor']
        )

        # Prob via sigmoid; center ~0.5 agar kalibrasi realistis
        probability = 1 / (1 + np.exp(-(risk_score - 0.5) * 8))
        probability = float(np.clip(probability, 0.01, 0.99))

        detailed_results = {
            'probability': probability,
            'risk_score': float(risk_score),
            'individual_risks': {
                'age': float(age_risk),
                'gender': float(gender_risk),
                'chest_pain': float(chest_pain_risk),
                'blood_pressure': float(bp_risk),
                'cholesterol': float(chol_risk),
                'diabetes': float(diabetes_risk),
                'ecg': float(ecg_risk),
                'exercise_capacity': float(exercise_capacity_risk),
                'exercise_angina': float(exercise_angina_risk),
                'st_depression': float(st_depression_risk),
                'st_slope': float(st_slope_risk)
            }
        }
        return detailed_results

# ================================================================================
# CLINICAL ANALYSIS FUNCTIONS
# ================================================================================

def analyze_risk_factors(patient_data):
    risk_factors, protective_factors = [], []

    # Age
    if patient_data['Age'] > 65:
        risk_factors.append(("Advanced age", "High", f"{patient_data['Age']} years"))
    elif patient_data['Age'] > 55:
        risk_factors.append(("Moderate age risk", "Medium", f"{patient_data['Age']} years"))
    else:
        protective_factors.append(("Young age", f"{patient_data['Age']} years"))

    # Gender
    if patient_data['Sex'] == 'M':
        risk_factors.append(("Male gender", "Medium", "Higher cardiovascular risk"))
    else:
        protective_factors.append(("Female gender", "Lower baseline risk"))

    # Blood pressure
    if patient_data['RestingBP'] >= 180:
        risk_factors.append(("Severe hypertension", "Very High", f"{patient_data['RestingBP']} mmHg"))
    elif patient_data['RestingBP'] >= 140:
        risk_factors.append(("Hypertension", "High", f"{patient_data['RestingBP']} mmHg"))
    elif patient_data['RestingBP'] >= 120:
        risk_factors.append(("Elevated blood pressure", "Medium", f"{patient_data['RestingBP']} mmHg"))
    else:
        protective_factors.append(("Normal blood pressure", f"{patient_data['RestingBP']} mmHg"))

    # Cholesterol
    if patient_data['Cholesterol'] >= 300:
        risk_factors.append(("Very high cholesterol", "Very High", f"{patient_data['Cholesterol']} mg/dl"))
    elif patient_data['Cholesterol'] >= 240:
        risk_factors.append(("High cholesterol", "High", f"{patient_data['Cholesterol']} mg/dl"))
    elif patient_data['Cholesterol'] >= 200:
        risk_factors.append(("Borderline cholesterol", "Medium", f"{patient_data['Cholesterol']} mg/dl"))
    else:
        protective_factors.append(("Optimal cholesterol", f"{patient_data['Cholesterol']} mg/dl"))

    # Diabetes
    if patient_data['FastingBS'] == 1:
        risk_factors.append(("Diabetes", "High", "Fasting BS > 120 mg/dl"))
    else:
        protective_factors.append(("Normal glucose", "Fasting BS ≤ 120 mg/dl"))

    # Exercise capacity
    predicted_max_hr = max(1.0, 220 - patient_data['Age'])
    hr_percentage = patient_data['MaxHR'] / predicted_max_hr
    if hr_percentage < 0.6:
        risk_factors.append(("Poor exercise capacity", "High", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))
    elif hr_percentage < 0.75:
        risk_factors.append(("Fair exercise capacity", "Medium", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))
    else:
        protective_factors.append(("Good exercise capacity", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))

    # Chest pain
    if patient_data['ChestPainType'] == 'ASY':
        risk_factors.append(("Asymptomatic presentation", "Very High", "Silent ischemia possible"))
    elif patient_data['ChestPainType'] == 'ATA':
        risk_factors.append(("Atypical angina", "Medium", "Unclear chest pain pattern"))

    # Exercise angina
    if patient_data['ExerciseAngina'] == 'Y':
        risk_factors.append(("Exercise-induced angina", "High", "Chest pain with exertion"))
    else:
        protective_factors.append(("No exercise angina", "No chest pain with exertion"))

    # ST depression
    if patient_data['Oldpeak'] >= 3.0:
        risk_factors.append(("Severe ST depression", "Very High", f"{patient_data['Oldpeak']} mm"))
    elif patient_data['Oldpeak'] >= 2.0:
        risk_factors.append(("Significant ST depression", "High", f"{patient_data['Oldpeak']} mm"))
    elif patient_data['Oldpeak'] >= 1.0:
        risk_factors.append(("Mild ST depression", "Medium", f"{patient_data['Oldpeak']} mm"))
    else:
        protective_factors.append(("No significant ST depression", f"{patient_data['Oldpeak']} mm"))

    # ECG
    if patient_data['RestingECG'] == 'LVH':
        risk_factors.append(("Left ventricular hypertrophy", "High", "Structural heart changes"))
    elif patient_data['RestingECG'] == 'ST':
        risk_factors.append(("ST-T wave abnormalities", "Medium", "Electrical conduction issues"))
    else:
        protective_factors.append(("Normal ECG", "No electrical abnormalities"))

    # ST slope
    if patient_data['ST_Slope'] == 'Down':
        risk_factors.append(("Downsloping ST segment", "High", "Significant ischemic response"))
    elif patient_data['ST_Slope'] == 'Flat':
        risk_factors.append(("Flat ST segment", "Medium", "Borderline ischemic response"))
    else:
        protective_factors.append(("Upsloping ST segment", "Normal exercise response"))

    return risk_factors, protective_factors

def generate_clinical_recommendations(risk_level, risk_probability, risk_factors):
    recommendations = {
        'immediate_actions': [],
        'diagnostic_tests': [],
        'lifestyle_modifications': [],
        'follow_up': [],
        'referrals': []
    }
    if risk_probability >= 0.8:
        recommendations['immediate_actions'] = [
            "Immediate cardiology consultation recommended",
            "Consider emergency cardiac evaluation if symptomatic",
            "Initiate cardiac monitoring if hospitalized",
            "Assess for acute coronary syndrome"
        ]
        recommendations['diagnostic_tests'] = [
            "Cardiac catheterization evaluation",
            "Stress echocardiography",
            "Cardiac CT angiography",
            "Complete lipid panel",
            "HbA1c and comprehensive metabolic panel"
        ]
        recommendations['follow_up'] = [
            "Cardiology follow-up within 1-2 weeks",
            "Blood pressure monitoring daily",
            "Symptom diary documentation"
        ]
    elif risk_probability >= 0.6:
        recommendations['immediate_actions'] = [
            "Urgent cardiology referral (within 2-4 weeks)",
            "Initiate cardiac risk reduction therapy",
            "Blood pressure optimization"
        ]
        recommendations['diagnostic_tests'] = [
            "Exercise or pharmacologic stress test",
            "Echocardiography",
            "Advanced lipid testing",
            "Coronary calcium scoring (if appropriate)"
        ]
        recommendations['follow_up'] = [
            "Primary care follow-up in 2-4 weeks",
            "Cardiology consultation within 4-6 weeks"
        ]
    elif risk_probability >= 0.4:
        recommendations['immediate_actions'] = [
            "Initiate cardiovascular risk reduction strategies",
            "Optimize blood pressure & cholesterol management",
            "Consider antiplatelet therapy (if appropriate)"
        ]
        recommendations['diagnostic_tests'] = [
            "Exercise stress test (if symptomatic)",
            "Basic echocardiography",
            "Complete cardiovascular risk assessment"
        ]
        recommendations['follow_up'] = [
            "Primary care follow-up in 6-8 weeks",
            "Annual cardiovascular risk reassessment"
        ]
    else:
        recommendations['immediate_actions'] = [
            "Continue preventive cardiovascular care",
            "Maintain healthy lifestyle habits"
        ]
        recommendations['diagnostic_tests'] = [
            "Routine annual physical examination",
            "Periodic cardiovascular risk screening"
        ]
        recommendations['follow_up'] = [
            "Annual primary care visits",
            "Cardiovascular risk reassessment every 3-5 years"
        ]

    # Universal lifestyle changes
    recommendations['lifestyle_modifications'] = [
        "Mediterranean-style diet with reduced saturated fat",
        "Regular aerobic exercise (150+ minutes/week)",
        "Smoking cessation (if applicable)",
        "Weight management to BMI 18.5-24.9",
        "Stress management techniques",
        "Adequate sleep (7-9 hours/night)",
        "Limited alcohol consumption"
    ]
    # Referrals
    if any("diabetes" in str(rf).lower() for rf in risk_factors):
        recommendations['referrals'].append("Endocrinology consultation for diabetes management")
    if any("hypertension" in str(rf).lower() or "blood pressure" in str(rf).lower() for rf in risk_factors):
        recommendations['referrals'].append("Consider nephrology if resistant hypertension")
    return recommendations

def calculate_prediction_confidence(detailed_results):
    risks = list(detailed_results['individual_risks'].values())
    hi = sum(1 for r in risks if r > 0.7)
    md = sum(1 for r in risks if 0.4 <= r <= 0.7)
    if hi >= 6:
        return "Very High Confidence"
    if hi >= 4 or (hi >= 2 and md >= 4):
        return "High Confidence"
    if md >= 6 or (hi >= 1 and md >= 3):
        return "Moderate Confidence"
    return "Lower Confidence - Mixed Factors"

def assess_clinical_significance(probability, detailed_results):
    if probability >= 0.7: return "Clinically significant risk requiring immediate intervention"
    if probability >= 0.5: return "Moderate clinical significance - intervention recommended"
    if probability >= 0.3: return "Some clinical significance - monitoring and lifestyle changes"
    return "Low clinical significance - continue preventive measures"

def interpret_prediction(probability, detailed_results):
    if probability >= 0.8:
        risk_level, risk_category, color, urg, timeline = "Very High Risk", "Critical", "#d32f2f", "Immediate medical attention required", "Within 24-48 hours"
    elif probability >= 0.6:
        risk_level, risk_category, color, urg, timeline = "High Risk", "Urgent", "#f57c00", "Prompt medical evaluation needed", "Within 1-2 weeks"
    elif probability >= 0.4:
        risk_level, risk_category, color, urg, timeline = "Moderate Risk", "Moderate", "#fbc02d", "Medical evaluation recommended", "Within 1-2 months"
    elif probability >= 0.2:
        risk_level, risk_category, color, urg, timeline = "Low-Moderate Risk", "Low-Moderate", "#689f38", "Routine medical follow-up", "Within 6 months"
    else:
        risk_level, risk_category, color, urg, timeline = "Low Risk", "Low", "#388e3c", "Continue preventive care", "Annual check-ups"
    return {
        'risk_level': risk_level,
        'risk_category': risk_category,
        'color': color,
        'urgency': urg,
        'timeline': timeline,
        'confidence_level': calculate_prediction_confidence(detailed_results),
        'clinical_significance': assess_clinical_significance(probability, detailed_results)
    }

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def get_risk_color(probability):
    if probability >= 0.8: return "#d32f2f"
    if probability >= 0.6: return "#f57c00"
    if probability >= 0.4: return "#fbc02d"
    if probability >= 0.2: return "#689f38"
    return "#388e3c"

def risk_css_class(probability):
    if probability >= 0.8: return "risk-very-high"
    if probability >= 0.6: return "risk-high"
    if probability >= 0.4: return "risk-moderate"
    return "risk-low"

def create_risk_gauge(probability):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Heart Disease Risk", 'font': {'size': 24, 'color': '#1f77b4'}},
        number={'font': {'size': 40, 'color': '#1f77b4'}},
        delta={'reference': 50, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': "darkblue", 'tickfont': {'size': 14}},
            'bar': {'color': get_risk_color(probability), 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#c8e6c9'},
                {'range': [20, 40], 'color': '#fff9c4'},
                {'range': [40, 60], 'color': '#ffcc80'},
                {'range': [60, 80], 'color': '#ffab91'},
                {'range': [80, 100], 'color': '#ef9a9a'}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
        }
    ))
    fig.update_layout(height=380, font={'color': "darkblue", 'family': "Arial"},
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig

def create_risk_factor_radar(detailed_results):
    individual_risks = detailed_results['individual_risks']
    categories = ['Age','Gender','Chest Pain','Blood Pressure','Cholesterol','Diabetes',
                  'ECG','Exercise Capacity','Exercise Angina','ST Depression','ST Slope']
    values = [
        individual_risks['age']*100, individual_risks['gender']*100, individual_risks['chest_pain']*100,
        individual_risks['blood_pressure']*100, individual_risks['cholesterol']*100, individual_risks['diabetes']*100,
        individual_risks['ecg']*100, individual_risks['exercise_capacity']*100, individual_risks['exercise_angina']*100,
        individual_risks['st_depression']*100, individual_risks['st_slope']*100
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=categories, fill='toself', name='Risk Factors',
        line_color='#1f77b4', fillcolor='rgba(31,119,180,0.3)'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10))),
        title="Individual Risk Factor Analysis", font=dict(size=12), height=460
    )
    return fig

def create_probability_distribution(probability):
    """
    Visualisasi distribusi ketidakpastian probabilitas menggunakan
    pendekatan Beta(a,b) tanpa SciPy (pakai math.gamma).
    """
    prob = float(np.clip(probability, 1e-3, 1-1e-3))
    k = 30.0  # konsentrasi (semakin besar -> kurva makin sempit)
    a = 1.0 + prob * k
    b = 1.0 + (1.0 - prob) * k

    xs = np.linspace(0.0, 1.0, 400)
    def beta_pdf(x, aa, bb):
        # x^{a-1}(1-x)^{b-1} / B(a,b)
        B = math.gamma(aa) * math.gamma(bb) / math.gamma(aa + bb)
        return np.where((x > 0) & (x < 1), (x**(aa-1) * (1-x)**(bb-1)) / B, 0.0)

    ys = beta_pdf(xs, a, b)

    # Normal approx untuk perkiraan 90% CI (tanpa SciPy)
    mean = a / (a + b)
    var = (a * b) / (((a + b)**2) * (a + b + 1.0))
    std = math.sqrt(var)
    lo = float(np.clip(mean - 1.645*std, 0.0, 1.0))
    hi = float(np.clip(mean + 1.645*std, 0.0, 1.0))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', name='Beta PDF', line=dict(width=3)))
    fig.add_vline(x=prob, line_dash="dash", line_color=get_risk_color(prob), annotation_text=f"Pred: {prob*100:.1f}%")
    fig.add_vrect(x0=lo, x1=hi, fillcolor="LightSkyBlue", opacity=0.25, line_width=0,
                  annotation_text=f"≈90% CI [{lo*100:.1f}%, {hi*100:.1f}%]", annotation_position="top left")

    fig.update_layout(title="Uncertainty of Risk Probability (Beta Approximation)",
                      xaxis_title="Probability", yaxis_title="Density",
                      height=380, showlegend=False)
    return fig

def create_top_contributors_bar(detailed_results, top_n=6):
    pairs = list(detailed_results['individual_risks'].items())
    # Konversi nama agar rapi
    label_map = {
        'age':'Age','gender':'Gender','chest_pain':'Chest Pain','blood_pressure':'Blood Pressure',
        'cholesterol':'Cholesterol','diabetes':'Diabetes','ecg':'ECG','exercise_capacity':'Exercise Capacity',
        'exercise_angina':'Exercise Angina','st_depression':'ST Depression','st_slope':'ST Slope'
    }
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_n]
    names = [label_map.get(k, k) for k,_ in top][::-1]
    vals = [v for _,v in top][::-1]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation='h'))
    fig.update_layout(title=f"Top {top_n} Risk Contributors", xaxis_title="Relative Risk (0-1)", height=360)
    return fig

# ================================================================================
# REPORT GENERATOR
# ================================================================================

def build_markdown_report(patient, prob, interp, risk_factors, protective_factors, recs):
    sex_word = "male" if patient['Sex'] == 'M' else "female"
    lines = []
    lines.append(f"# Heart Disease Prediction Report\nGenerated: {datetime.utcnow().isoformat()} UTC\n")
    lines.append("## Patient Profile")
    lines.append(f"- Age: **{patient['Age']}** years")
    lines.append(f"- Sex: **{sex_word}**")
    lines.append(f"- Chest Pain Type: **{patient['ChestPainType']}**")
    lines.append(f"- Resting BP: **{patient['RestingBP']} mmHg**")
    lines.append(f"- Cholesterol: **{patient['Cholesterol']} mg/dl**")
    lines.append(f"- Fasting BS >120: **{patient['FastingBS']}**")
    lines.append(f"- Resting ECG: **{patient['RestingECG']}**")
    lines.append(f"- Max HR: **{patient['MaxHR']} bpm**")
    lines.append(f"- Exercise Angina: **{patient['ExerciseAngina']}**")
    lines.append(f"- Oldpeak: **{patient['Oldpeak']}**")
    lines.append(f"- ST Slope: **{patient['ST_Slope']}**")
    lines.append("\n## Prediction")
    lines.append(f"- **Risk Level:** {interp['risk_level']}  ")
    lines.append(f"- **Probability:** {prob*100:.1f}%  ")
    lines.append(f"- **Confidence:** {interp['confidence_level']}  ")
    lines.append(f"- **Clinical Significance:** {interp['clinical_significance']}  ")
    lines.append(f"- **Urgency:** {interp['urgency']} ({interp['timeline']})")
    lines.append("\n## Risk Factors")
    if risk_factors:
        for name, sev, note in risk_factors:
            lines.append(f"- **{name}** — {sev} ({note})")
    else:
        lines.append("- None prominent")
    lines.append("\n## Protective Factors")
    if protective_factors:
        for p in protective_factors:
            if isinstance(p, tuple):
                lines.append(f"- {p[0]} — {p[1] if len(p)>1 else ''}")
            else:
                lines.append(f"- {p}")
    else:
        lines.append("- Not identified")

    lines.append("\n## Recommendations")
    for k,v in recs.items():
        title = k.replace('_',' ').title()
        if v:
            lines.append(f"### {title}")
            for item in v:
                lines.append(f"- {item}")
    lines.append("\n> **Medical Disclaimer:** This report is informational and does not replace professional medical advice.")
    return "\n".join(lines)

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
exercise_angina = st.sidebar.selectbox("Exercise Induced Angina", ["N", "Y"])

st.sidebar.markdown("### Vital Signs & Laboratory")
resting_bp = st.sidebar.number_input("Resting Blood Pressure (mmHg)", min_value=80, max_value=220, value=120, step=1)
cholesterol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200, step=1)
fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])

st.sidebar.markdown("### Cardiac Test Results")
resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150, step=1)
oldpeak = st.sidebar.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=6.0, value=1.0, step=0.1)
st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])

predict_button = st.sidebar.button("🔍 Analyze Heart Disease Risk", type="primary")

# ================================================================================
# MAIN CONTENT
# ================================================================================

st.markdown('<p class="main-header">❤️ Heart Disease Prediction System</p>', unsafe_allow_html=True)
st.markdown("### Advanced clinical decision support with transparent risk explanation")

with st.expander("ℹ️ About This App", expanded=False):
    st.markdown("""
    - **Model**: rules-based clinical scorer → *tidak memerlukan file model eksternal*, stabil untuk deployment.
    - **Output**: probabilitas risiko, faktor kontribusi, dan rekomendasi klinis.
    - **Catatan**: Ini bukan alat diagnosis; keputusan klinis tetap pada tenaga medis.
    """)

if predict_button:
    patient = {
        "Age": age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
        "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
        "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope
    }

    predictor = AdvancedHeartDiseasePredictor()
    detailed = predictor.predict_risk_probability(patient)
    prob = detailed['probability']

    risk_factors, protective_factors = analyze_risk_factors(patient)
    interp = interpret_prediction(prob, detailed)
    recs = generate_clinical_recommendations(interp['risk_level'], prob, risk_factors)

    # ======= Results Section =======
    st.markdown("---")
    st.markdown('<p class="sub-header">🎯 Prediction Results</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,3])
    with c1:
        st.markdown(f"""
        <div class="metric-card {risk_css_class(prob)}">
            <h3 style="margin:0;color:{interp['color']};">Risk Level: {interp['risk_level']}</h3>
            <h2 style="margin:10px 0;color:{interp['color']};">{prob*100:.1f}%</h2>
            <p style="margin:0;">Probability of Heart Disease</p>
            <p style="margin:0;font-size:13px;color:#444;">Confidence: {interp['confidence_level']}</p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.plotly_chart(create_risk_gauge(prob), use_container_width=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin:0;">Clinical Guidance</h3>
            <p style="margin:8px 0 0 0;"><b>Significance:</b> {interp['clinical_significance']}</p>
            <p style="margin:0;"><b>Urgency:</b> {interp['urgency']}</p>
            <p style="margin:0;"><b>Suggested timeline:</b> {interp['timeline']}</p>
        </div>
        """, unsafe_allow_html=True)

    # Visualizations
    st.markdown("### 📊 Analytical Views")
    v1, v2 = st.columns(2)
    with v1:
        st.plotly_chart(create_risk_factor_radar(detailed), use_container_width=True)
    with v2:
        st.plotly_chart(create_probability_distribution(prob), use_container_width=True)

    st.plotly_chart(create_top_contributors_bar(detailed, top_n=6), use_container_width=True)

    # Factors table
    st.markdown("### ⚠️ Risk & ✅ Protective Factors")
    t1, t2 = st.columns(2)
    with t1:
        df_risk = pd.DataFrame(risk_factors, columns=["Factor","Severity","Details"]) if risk_factors else pd.DataFrame(columns=["Factor","Severity","Details"])
        st.dataframe(df_risk, use_container_width=True, height=min(400, 40*(len(df_risk)+1)))
    with t2:
        # Protective factor bisa berupa tuple atau string; normalkan
        prot_rows = []
        for p in protective_factors:
            if isinstance(p, tuple):
                prot_rows.append({"Protective Factor": p[0], "Details": p[1] if len(p)>1 else ""})
            else:
                prot_rows.append({"Protective Factor": str(p), "Details": ""})
        df_prot = pd.DataFrame(prot_rows) if prot_rows else pd.DataFrame(columns=["Protective Factor","Details"])
        st.dataframe(df_prot, use_container_width=True, height=min(400, 40*(len(df_prot)+1)))

    # Recommendations
    st.markdown("### 🩺 Recommendations")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Immediate / Short-term Actions**")
        st.markdown("\n".join([f"- {x}" for x in recs['immediate_actions']]) or "- None")
        st.markdown("**Diagnostic Tests**")
        st.markdown("\n".join([f"- {x}" for x in recs['diagnostic_tests']]) or "- None")
        st.markdown("**Follow-up**")
        st.markdown("\n".join([f"- {x}" for x in recs['follow_up']]) or "- None")
    with rc2:
        st.markdown("**Lifestyle Modifications**")
        st.markdown("\n".join([f"- {x}" for x in recs['lifestyle_modifications']]) or "- None")
        st.markdown("**Referrals**")
        st.markdown("\n".join([f"- {x}" for x in recs['referrals']]) or "- None")

    # Download report
    st.markdown("### 📄 Downloadable Report")
    report_md = build_markdown_report(patient, prob, interp, risk_factors, protective_factors, recs)
    st.download_button("⬇️ Download Report (Markdown)", report_md.encode("utf-8"),
                       file_name=f"heart_risk_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md",
                       mime="text/markdown")

    # Disclaimer
    st.markdown("---")
    st.warning("**Medical Disclaimer:** This tool is for informational purposes only and does not replace professional medical advice.")

else:
    st.markdown("### 🚀 Welcome")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **How to Use:**
        1. Isi data pasien di sidebar  
        2. Klik **Analyze Heart Disease Risk**  
        3. Tinjau hasil, faktor risiko, dan rekomendasi
        """)
    with c2:
        st.markdown("""
        **Clinical Parameters:**
        - Demographics: Age, Sex
        - Symptoms: Chest pain, Exercise angina
        - Vitals/Labs: BP, Cholesterol, Blood sugar
        - Tests: ECG, Max HR, ST slope
        """)

# ================================================================================
# FOOTER
# ================================================================================

st.markdown("---")
st.markdown("""
<div style="text-align:center;color:#666;padding:12px;">
  <p><strong>Heart Disease Prediction System</strong> | Transparent, explainable, and clinician-friendly</p>
  <p>Version 1.0 • Rules-based engine for stable deployment</p>
</div>
""", unsafe_allow_html=True)
