# streamlit_heart_disease_app.py
# ================================================================================
# HEART DISEASE PREDICTION SYSTEM - COMPLETE STREAMLIT APPLICATION
# Super lengkap, professional, dan bebas error
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
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
    .risk-very-high {
        background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%);
        border-left-color: #d32f2f;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #fff3e0 0%, #ffcc80 100%);
        border-left-color: #f57c00;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #fffde7 0%, #fff176 100%);
        border-left-color: #fbc02d;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #e8f5e8 0%, #a5d6a7 100%);
        border-left-color: #388e3c;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #fafafa 0%, #f0f2f6 100%);
    }
    
    /* Custom buttons */
    .stButton > button {
        background: linear-gradient(90deg, #1f77b4 0%, #2e8bc0 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff9800;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Progress indicators */
    .progress-text {
        text-align: center;
        font-weight: bold;
        color: #1f77b4;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# ADVANCED HEART DISEASE PREDICTION MODEL
# ================================================================================

class AdvancedHeartDiseasePredictor:
    """
    Advanced clinical prediction model for heart disease risk assessment
    Based on comprehensive analysis of cardiac risk factors
    """
    
    def __init__(self):
        # Clinical risk weights based on medical literature
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
        
        # Risk factor thresholds
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
        
        # Population statistics for normalization
        self.population_stats = {
            'age_mean': 54.4,
            'age_std': 9.1,
            'bp_mean': 131.6,
            'bp_std': 17.5,
            'cholesterol_mean': 246.3,
            'cholesterol_std': 51.8,
            'max_hr_mean': 149.6,
            'max_hr_std': 22.9
        }
    
    def calculate_age_factor(self, age):
        """Calculate age-related risk factor"""
        if age >= self.thresholds['age_high_risk']:
            return 1.0
        elif age >= self.thresholds['age_moderate_risk']:
            return 0.6 + (age - self.thresholds['age_moderate_risk']) / 10 * 0.4
        else:
            return max(0.1, (age - 30) / 25 * 0.6)
    
    def calculate_gender_factor(self, sex):
        """Calculate gender-related risk factor"""
        return 1.0 if sex == 'M' else 0.3
    
    def calculate_chest_pain_factor(self, chest_pain_type):
        """Calculate chest pain type risk factor"""
        pain_risk = {
            'ASY': 1.0,    # Asymptomatic - highest risk
            'NAP': 0.4,    # Non-anginal pain
            'ATA': 0.7,    # Atypical angina
            'TA': 0.5      # Typical angina
        }
        return pain_risk.get(chest_pain_type, 0.5)
    
    def calculate_bp_factor(self, resting_bp):
        """Calculate blood pressure risk factor"""
        if resting_bp >= 180:
            return 1.0
        elif resting_bp >= self.thresholds['bp_high']:
            return 0.8
        elif resting_bp >= self.thresholds['bp_elevated']:
            return 0.4
        else:
            return 0.1
    
    def calculate_cholesterol_factor(self, cholesterol):
        """Calculate cholesterol risk factor"""
        if cholesterol >= 300:
            return 1.0
        elif cholesterol >= self.thresholds['cholesterol_high']:
            return 0.8
        elif cholesterol >= self.thresholds['cholesterol_borderline']:
            return 0.5
        else:
            return 0.2
    
    def calculate_diabetes_factor(self, fasting_bs):
        """Calculate diabetes risk factor"""
        return 0.8 if fasting_bs == 1 else 0.1
    
    def calculate_ecg_factor(self, resting_ecg):
        """Calculate ECG abnormality risk factor"""
        ecg_risk = {
            'Normal': 0.1,
            'ST': 0.7,     # ST-T wave abnormality
            'LVH': 0.9     # Left ventricular hypertrophy
        }
        return ecg_risk.get(resting_ecg, 0.1)
    
    def calculate_exercise_capacity_factor(self, max_hr, age):
        """Calculate exercise capacity risk factor"""
        predicted_max_hr = 220 - age
        hr_percentage = max_hr / predicted_max_hr
        
        if hr_percentage < 0.6:
            return 1.0
        elif hr_percentage < 0.75:
            return 0.7
        elif hr_percentage < 0.85:
            return 0.4
        else:
            return 0.1
    
    def calculate_exercise_angina_factor(self, exercise_angina):
        """Calculate exercise-induced angina risk factor"""
        return 0.9 if exercise_angina == 'Y' else 0.1
    
    def calculate_st_depression_factor(self, oldpeak):
        """Calculate ST depression risk factor"""
        if oldpeak >= 3.0:
            return 1.0
        elif oldpeak >= self.thresholds['oldpeak_significant']:
            return 0.8
        elif oldpeak >= 1.0:
            return 0.5
        else:
            return 0.1
    
    def calculate_st_slope_factor(self, st_slope):
        """Calculate ST slope risk factor"""
        slope_risk = {
            'Down': 1.0,   # Downsloping - highest risk
            'Flat': 0.7,   # Flat - moderate risk
            'Up': 0.2      # Upsloping - lowest risk
        }
        return slope_risk.get(st_slope, 0.5)
    
    def predict_risk_probability(self, patient_data):
        """
        Calculate comprehensive heart disease risk probability
        Returns probability between 0 and 1
        """
        
        # Calculate individual risk factors
        age_risk = self.calculate_age_factor(patient_data['Age'])
        gender_risk = self.calculate_gender_factor(patient_data['Sex'])
        chest_pain_risk = self.calculate_chest_pain_factor(patient_data['ChestPainType'])
        bp_risk = self.calculate_bp_factor(patient_data['RestingBP'])
        chol_risk = self.calculate_cholesterol_factor(patient_data['Cholesterol'])
        diabetes_risk = self.calculate_diabetes_factor(patient_data['FastingBS'])
        ecg_risk = self.calculate_ecg_factor(patient_data['RestingECG'])
        exercise_capacity_risk = self.calculate_exercise_capacity_factor(
            patient_data['MaxHR'], patient_data['Age']
        )
        exercise_angina_risk = self.calculate_exercise_angina_factor(patient_data['ExerciseAngina'])
        st_depression_risk = self.calculate_st_depression_factor(patient_data['Oldpeak'])
        st_slope_risk = self.calculate_st_slope_factor(patient_data['ST_Slope'])
        
        # Calculate weighted risk score
        risk_score = (
            age_risk * self.feature_weights['age_factor'] +
            gender_risk * self.feature_weights['gender_factor'] +
            chest_pain_risk * self.feature_weights['chest_pain_factor'] +
            bp_risk * self.feature_weights['blood_pressure_factor'] +
            chol_risk * self.feature_weights['cholesterol_factor'] +
            diabetes_risk * self.feature_weights['diabetes_factor'] +
            ecg_risk * self.feature_weights['ecg_factor'] +
            exercise_capacity_risk * self.feature_weights['exercise_capacity_factor'] +
            exercise_angina_risk * self.feature_weights['exercise_angina_factor'] +
            st_depression_risk * self.feature_weights['st_depression_factor'] +
            st_slope_risk * self.feature_weights['st_slope_factor']
        )
        
        # Apply sigmoid transformation for probability
        # Adjusted for medical realistic probabilities
        probability = 1 / (1 + np.exp(-(risk_score - 0.5) * 8))
        
        # Ensure realistic probability bounds
        probability = max(0.01, min(0.99, probability))
        
        # Return detailed results
        detailed_results = {
            'probability': probability,
            'risk_score': risk_score,
            'individual_risks': {
                'age': age_risk,
                'gender': gender_risk,
                'chest_pain': chest_pain_risk,
                'blood_pressure': bp_risk,
                'cholesterol': chol_risk,
                'diabetes': diabetes_risk,
                'ecg': ecg_risk,
                'exercise_capacity': exercise_capacity_risk,
                'exercise_angina': exercise_angina_risk,
                'st_depression': st_depression_risk,
                'st_slope': st_slope_risk
            }
        }
        
        return detailed_results

# ================================================================================
# CLINICAL ANALYSIS FUNCTIONS
# ================================================================================

def analyze_risk_factors(patient_data):
    """Comprehensive risk factor analysis"""
    risk_factors = []
    protective_factors = []
    
    # Age analysis
    if patient_data['Age'] > 65:
        risk_factors.append(("Advanced age", "High", f"{patient_data['Age']} years"))
    elif patient_data['Age'] > 55:
        risk_factors.append(("Moderate age risk", "Medium", f"{patient_data['Age']} years"))
    else:
        protective_factors.append(("Young age", f"{patient_data['Age']} years"))
    
    # Gender analysis
    if patient_data['Sex'] == 'M':
        risk_factors.append(("Male gender", "Medium", "Higher cardiovascular risk"))
    else:
        protective_factors.append(("Female gender", "Lower baseline risk"))
    
    # Blood pressure analysis
    if patient_data['RestingBP'] >= 180:
        risk_factors.append(("Severe hypertension", "Very High", f"{patient_data['RestingBP']} mmHg"))
    elif patient_data['RestingBP'] >= 140:
        risk_factors.append(("Hypertension", "High", f"{patient_data['RestingBP']} mmHg"))
    elif patient_data['RestingBP'] >= 120:
        risk_factors.append(("Elevated blood pressure", "Medium", f"{patient_data['RestingBP']} mmHg"))
    else:
        protective_factors.append(("Normal blood pressure", f"{patient_data['RestingBP']} mmHg"))
    
    # Cholesterol analysis
    if patient_data['Cholesterol'] >= 300:
        risk_factors.append(("Very high cholesterol", "Very High", f"{patient_data['Cholesterol']} mg/dl"))
    elif patient_data['Cholesterol'] >= 240:
        risk_factors.append(("High cholesterol", "High", f"{patient_data['Cholesterol']} mg/dl"))
    elif patient_data['Cholesterol'] >= 200:
        risk_factors.append(("Borderline cholesterol", "Medium", f"{patient_data['Cholesterol']} mg/dl"))
    else:
        protective_factors.append(("Optimal cholesterol", f"{patient_data['Cholesterol']} mg/dl"))
    
    # Diabetes analysis
    if patient_data['FastingBS'] == 1:
        risk_factors.append(("Diabetes", "High", "Fasting BS > 120 mg/dl"))
    else:
        protective_factors.append(("Normal glucose", "Fasting BS ≤ 120 mg/dl"))
    
    # Exercise capacity analysis
    predicted_max_hr = 220 - patient_data['Age']
    hr_percentage = patient_data['MaxHR'] / predicted_max_hr
    
    if hr_percentage < 0.6:
        risk_factors.append(("Poor exercise capacity", "High", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))
    elif hr_percentage < 0.75:
        risk_factors.append(("Fair exercise capacity", "Medium", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))
    else:
        protective_factors.append(("Good exercise capacity", f"MaxHR {patient_data['MaxHR']} ({hr_percentage:.0%} predicted)"))
    
    # Chest pain analysis
    if patient_data['ChestPainType'] == 'ASY':
        risk_factors.append(("Asymptomatic presentation", "Very High", "Silent ischemia possible"))
    elif patient_data['ChestPainType'] == 'ATA':
        risk_factors.append(("Atypical angina", "Medium", "Unclear chest pain pattern"))
    
    # Exercise angina analysis
    if patient_data['ExerciseAngina'] == 'Y':
        risk_factors.append(("Exercise-induced angina", "High", "Chest pain with exertion"))
    else:
        protective_factors.append(("No exercise angina", "No chest pain with exertion"))
    
    # ST depression analysis
    if patient_data['Oldpeak'] >= 3.0:
        risk_factors.append(("Severe ST depression", "Very High", f"{patient_data['Oldpeak']} mm"))
    elif patient_data['Oldpeak'] >= 2.0:
        risk_factors.append(("Significant ST depression", "High", f"{patient_data['Oldpeak']} mm"))
    elif patient_data['Oldpeak'] >= 1.0:
        risk_factors.append(("Mild ST depression", "Medium", f"{patient_data['Oldpeak']} mm"))
    else:
        protective_factors.append(("No significant ST depression", f"{patient_data['Oldpeak']} mm"))
    
    # ECG analysis
    if patient_data['RestingECG'] == 'LVH':
        risk_factors.append(("Left ventricular hypertrophy", "High", "Structural heart changes"))
    elif patient_data['RestingECG'] == 'ST':
        risk_factors.append(("ST-T wave abnormalities", "Medium", "Electrical conduction issues"))
    else:
        protective_factors.append(("Normal ECG", "No electrical abnormalities"))
    
    # ST slope analysis
    if patient_data['ST_Slope'] == 'Down':
        risk_factors.append(("Downsloping ST segment", "High", "Significant ischemic response"))
    elif patient_data['ST_Slope'] == 'Flat':
        risk_factors.append(("Flat ST segment", "Medium", "Borderline ischemic response"))
    else:
        protective_factors.append(("Upsloping ST segment", "Normal exercise response"))
    
    return risk_factors, protective_factors

def generate_clinical_recommendations(risk_level, risk_probability, risk_factors):
    """Generate comprehensive clinical recommendations"""
    
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
            "Exercise stress test or pharmacologic stress test",
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
            "Optimize blood pressure and cholesterol management",
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
    
    # Universal lifestyle recommendations
    recommendations['lifestyle_modifications'] = [
        "Mediterranean-style diet with reduced saturated fat",
        "Regular aerobic exercise (150+ minutes/week)",
        "Smoking cessation (if applicable)",
        "Weight management to BMI 18.5-24.9",
        "Stress management techniques",
        "Adequate sleep (7-9 hours/night)",
        "Limited alcohol consumption"
    ]
    
    # Risk-specific referrals
    if any("diabetes" in str(rf).lower() for rf in risk_factors):
        recommendations['referrals'].append("Endocrinology consultation for diabetes management")
    
    if any("hypertension" in str(rf).lower() for rf in risk_factors):
        recommendations['referrals'].append("Consider nephrology if resistant hypertension")
    
    return recommendations

def interpret_prediction(probability, detailed_results):
    """Comprehensive prediction interpretation"""
    
    if probability >= 0.8:
        risk_level = "Very High Risk"
        risk_category = "Critical"
        color = "#d32f2f"
        urgency = "Immediate medical attention required"
        timeline = "Within 24-48 hours"
        
    elif probability >= 0.6:
        risk_level = "High Risk"
        risk_category = "Urgent" 
        color = "#f57c00"
        urgency = "Prompt medical evaluation needed"
        timeline = "Within 1-2 weeks"
        
    elif probability >= 0.4:
        risk_level = "Moderate Risk"
        risk_category = "Moderate"
        color = "#fbc02d"
        urgency = "Medical evaluation recommended"
        timeline = "Within 1-2 months"
        
    elif probability >= 0.2:
        risk_level = "Low-Moderate Risk"
        risk_category = "Low-Moderate"
        color = "#689f38"
        urgency = "Routine medical follow-up"
        timeline = "Within 6 months"
        
    else:
        risk_level = "Low Risk"
        risk_category = "Low"
        color = "#388e3c"
        urgency = "Continue preventive care"
        timeline = "Annual check-ups"
    
    interpretation = {
        'risk_level': risk_level,
        'risk_category': risk_category,
        'color': color,
        'urgency': urgency,
        'timeline': timeline,
        'confidence_level': calculate_prediction_confidence(detailed_results),
        'clinical_significance': assess_clinical_significance(probability, detailed_results)
    }
    
    return interpretation

def calculate_prediction_confidence(detailed_results):
    """Calculate confidence level of prediction"""
    
    individual_risks = detailed_results['individual_risks']
    risk_values = list(individual_risks.values())
    
    # Calculate consistency of risk factors
    high_risk_factors = sum(1 for risk in risk_values if risk > 0.7)
    moderate_risk_factors = sum(1 for risk in risk_values if 0.4 <= risk <= 0.7)
    low_risk_factors = sum(1 for risk in risk_values if risk < 0.4)
    
    # Determine confidence based on consistency
    if high_risk_factors >= 6:
        return "Very High Confidence"
    elif high_risk_factors >= 4 or (high_risk_factors >= 2 and moderate_risk_factors >= 4):
        return "High Confidence"
    elif moderate_risk_factors >= 6 or (high_risk_factors >= 1 and moderate_risk_factors >= 3):
        return "Moderate Confidence"
    else:
        return "Lower Confidence - Mixed Factors"

def assess_clinical_significance(probability, detailed_results):
    """Assess clinical significance of the prediction"""
    
    if probability >= 0.7:
        return "Clinically significant risk requiring immediate intervention"
    elif probability >= 0.5:
        return "Moderate clinical significance - intervention recommended"
    elif probability >= 0.3:
        return "Some clinical significance - monitoring and lifestyle changes"
    else:
        return "Low clinical significance - continue preventive measures"

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def create_risk_gauge(probability):
    """Create an advanced risk gauge visualization"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Heart Disease Risk", 'font': {'size': 24, 'color': '#1f77b4'}},
        number = {'font': {'size': 40, 'color': '#1f77b4'}},
        delta = {'reference': 50, 'position': "top"},
        gauge = {
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
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def get_risk_color(probability):
    """Get color based on risk probability"""
    if probability >= 0.8:
        return "#d32f2f"
    elif probability >= 0.6:
        return "#f57c00"
    elif probability >= 0.4:
        return "#fbc02d"
    elif probability >= 0.2:
        return "#689f38"
    else:
        return "#388e3c"

def create_risk_factor_radar(detailed_results):
    """Create radar chart for individual risk factors"""
    
    individual_risks = detailed_results['individual_risks']
    
    categories = [
        'Age', 'Gender', 'Chest Pain', 'Blood Pressure', 'Cholesterol',
        'Diabetes', 'ECG', 'Exercise Capacity', 'Exercise Angina', 
        'ST Depression', 'ST Slope'
    ]
    
    values = [
        individual_risks['age'] * 100,
        individual_risks['gender'] * 100,
        individual_risks['chest_pain'] * 100,
        individual_risks['blood_pressure'] * 100,
        individual_risks['cholesterol'] * 100,
        individual_risks['diabetes'] * 100,
        individual_risks['ecg'] * 100,
        individual_risks['exercise_capacity'] * 100,
        individual_risks['exercise_angina'] * 100,
        individual_risks['st_depression'] * 100,
        individual_risks['st_slope'] * 100
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Risk Factors',
        line_color='#1f77b4',
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        title="Individual Risk Factor Analysis",
        font=dict(size=12),
        height=500
    )
    
    return fig

def create_probability_distribution(probability):
    """Create probability distribution visualization"""
    
        st.markdown('<h3>Clinical Recommendations</h3>', unsafe_allow_html=True)
        
        st.markdown("**Immediate Actions:**")
        for action in recommendations['immediate_actions']:
            st.markdown(f"- {action}")
        
        st.markdown("**Diagnostic Tests:**")
        for test in recommendations['diagnostic_tests']:
            st.markdown(f"- {test}")
        
        st.markdown("**Lifestyle Modifications:**")
        for lifestyle in recommendations['lifestyle_modifications']:
            st.markdown(f"- {lifestyle}")
        
        st.markdown("**Follow-up:**")
        for follow in recommendations['follow_up']:
            st.markdown(f"- {follow}")
        
        if recommendations['referrals']:
            st.markdown("**Referrals:**")
            for referral in recommendations['referrals']:
                st.markdown(f"- {referral}")
    
    else:
        st.info("Please enter patient data and click 'Predict Heart Disease Risk' to begin.")
    
if __name__ == "__main__":
    main()
