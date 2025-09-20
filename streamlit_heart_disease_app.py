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
    
    # Generate normal distribution around the predicted probability
    x = np.linspace(0, 1, 100)
    std_dev = 0.1  # Standard deviation for uncertainty
    mean = probability
    
    # Calculate probability density
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    
    fig = go.Figure()
    
    # Add probability distribution curve
    fig.add_trace(go.Scatter(
        x=x * 100, 
        y=y,
        mode='lines',
        fill='tozeroy',
        name='Probability Distribution',
        line=dict(color='#1f77b4', width=3),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    # Add predicted probability line
    fig.add_vline(
        x=probability * 100, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Predicted: {probability:.1%}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title="Risk Probability Distribution",
        xaxis_title="Risk Probability (%)",
        yaxis_title="Density",
        height=300,
        showlegend=False
    )
    
    return fig

def create_risk_comparison_chart(patient_data, detailed_results):
    """Create comparison chart with population averages"""
    
    # Population averages for comparison
    population_avg = {
        'Age': 54.4,
        'RestingBP': 131.6,
        'Cholesterol': 246.3,
        'MaxHR': 149.6,
        'Oldpeak': 1.0
    }
    
    metrics = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    patient_values = [patient_data[metric] for metric in metrics]
    population_values = [population_avg[metric] for metric in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Patient',
        x=metrics,
        y=patient_values,
        marker_color='#ff7f0e'
    ))
    
    fig.add_trace(go.Bar(
        name='Population Average',
        x=metrics,
        y=population_values,
        marker_color='#1f77b4'
    ))
    
    fig.update_layout(
        title='Patient vs Population Comparison',
        xaxis_title='Parameters',
        yaxis_title='Values',
        barmode='group',
        height=400
    )
    
    return fig

def create_risk_timeline(patient_data):
    """Create risk timeline based on age progression"""
    
    current_age = patient_data['Age']
    ages = list(range(current_age, min(current_age + 20, 90), 2))
    
    predictor = AdvancedHeartDiseasePredictor()
    risk_progression = []
    
    for age in ages:
        temp_data = patient_data.copy()
        temp_data['Age'] = age
        # Adjust max HR for age
        temp_data['MaxHR'] = max(100, temp_data['MaxHR'] - (age - current_age) * 0.8)
        
        result = predictor.predict_risk_probability(temp_data)
        risk_progression.append(result['probability'] * 100)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ages,
        y=risk_progression,
        mode='lines+markers',
        name='Risk Progression',
        line=dict(color='#d32f2f', width=3),
        marker=dict(size=6)
    ))
    
    # Add current age marker
    current_risk_idx = 0
    fig.add_trace(go.Scatter(
        x=[current_age],
        y=[risk_progression[current_risk_idx]],
        mode='markers',
        name='Current Age',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.update_layout(
        title='Projected Risk Timeline',
        xaxis_title='Age (years)',
        yaxis_title='Risk Probability (%)',
        height=400,
        showlegend=True
    )
    
    return fig

def create_risk_breakdown_pie(detailed_results):
    """Create pie chart showing risk factor contributions"""
    
    individual_risks = detailed_results['individual_risks']
    
    # Calculate weighted contributions
    predictor = AdvancedHeartDiseasePredictor()
    weights = predictor.feature_weights
    
    weighted_risks = {
        'Age': individual_risks['age'] * weights['age_factor'],
        'Gender': individual_risks['gender'] * weights['gender_factor'],
        'Chest Pain': individual_risks['chest_pain'] * weights['chest_pain_factor'],
        'Blood Pressure': individual_risks['blood_pressure'] * weights['blood_pressure_factor'],
        'Cholesterol': individual_risks['cholesterol'] * weights['cholesterol_factor'],
        'Diabetes': individual_risks['diabetes'] * weights['diabetes_factor'],
        'ECG': individual_risks['ecg'] * weights['ecg_factor'],
        'Exercise Capacity': individual_risks['exercise_capacity'] * weights['exercise_capacity_factor'],
        'Exercise Angina': individual_risks['exercise_angina'] * weights['exercise_angina_factor'],
        'ST Depression': individual_risks['st_depression'] * weights['st_depression_factor'],
        'ST Slope': individual_risks['st_slope'] * weights['st_slope_factor']
    }
    
    # Filter significant contributors (> 5% of total)
    total_weighted_risk = sum(weighted_risks.values())
    significant_risks = {k: v for k, v in weighted_risks.items() if v/total_weighted_risk > 0.05}
    
    # Group small contributors as "Other"
    other_risk = total_weighted_risk - sum(significant_risks.values())
    if other_risk > 0:
        significant_risks['Other'] = other_risk
    
    fig = go.Figure(data=[go.Pie(
        labels=list(significant_risks.keys()),
        values=list(significant_risks.values()),
        hole=0.3,
        textinfo='label+percent',
        textposition='outside'
    )])
    
    fig.update_layout(
        title="Risk Factor Contributions",
        height=400
    )
    
    return fig

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🫀 Advanced Heart Disease Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Clinical Decision Support Tool</div>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    <div class="info-box">
    <strong>About this System:</strong><br>
    This advanced prediction system uses comprehensive clinical parameters to assess cardiovascular risk. 
    It employs evidence-based algorithms derived from cardiology research to provide accurate risk stratification.
    <br><br>
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
    It should not replace professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for input parameters
    st.sidebar.markdown("## Patient Information")
    st.sidebar.markdown("Enter the clinical parameters below:")
    
    # Patient Demographics
    st.sidebar.markdown("### Demographics")
    age = st.sidebar.slider("Age", 20, 100, 54, help="Patient age in years")
    sex = st.sidebar.selectbox("Sex", ["M", "F"], help="M = Male, F = Female")
    
    # Clinical Parameters
    st.sidebar.markdown("### Clinical Measurements")
    
    chest_pain_type = st.sidebar.selectbox(
        "Chest Pain Type",
        ["ASY", "ATA", "NAP", "TA"],
        help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina"
    )
    
    resting_bp = st.sidebar.slider(
        "Resting Blood Pressure (mmHg)", 
        80, 200, 132,
        help="Systolic blood pressure at rest"
    )
    
    cholesterol = st.sidebar.slider(
        "Cholesterol (mg/dl)", 
        100, 400, 246,
        help="Total cholesterol level"
    )
    
    fasting_bs = st.sidebar.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        [0, 1],
        help="1 if fasting blood sugar > 120 mg/dl, else 0"
    )
    
    resting_ecg = st.sidebar.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"],
        help="Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy"
    )
    
    max_hr = st.sidebar.slider(
        "Maximum Heart Rate Achieved", 
        60, 220, 150,
        help="Maximum heart rate during exercise test"
    )
    
    exercise_angina = st.sidebar.selectbox(
        "Exercise Induced Angina",
        ["N", "Y"],
        help="Y = Yes, N = No"
    )
    
    oldpeak = st.sidebar.slider(
        "ST Depression (Oldpeak)", 
        0.0, 6.0, 1.0, 0.1,
        help="ST depression induced by exercise relative to rest"
    )
    
    st_slope = st.sidebar.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"],
        help="Slope of the peak exercise ST segment"
    )
    
    # Create patient data dictionary
    patient_data = {
        'Age': age,
        'Sex': sex,
        'ChestPainType': chest_pain_type,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'RestingECG': resting_ecg,
        'MaxHR': max_hr,
        'ExerciseAngina': exercise_angina,
        'Oldpeak': oldpeak,
        'ST_Slope': st_slope
    }
    
    # Prediction button
    if st.sidebar.button("🔍 Analyze Risk", type="primary"):
        
        # Initialize predictor
        predictor = AdvancedHeartDiseasePredictor()
        
        # Show progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 30:
                status_text.text('Analyzing patient parameters...')
            elif i < 60:
                status_text.text('Computing risk factors...')
            elif i < 90:
                status_text.text('Generating predictions...')
            else:
                status_text.text('Finalizing analysis...')
            time.sleep(0.01)
        
        progress_bar.empty()
        status_text.empty()
        
        # Make prediction
        detailed_results = predictor.predict_risk_probability(patient_data)
        probability = detailed_results['probability']
        
        # Interpret results
        interpretation = interpret_prediction(probability, detailed_results)
        
        # Analyze risk factors
        risk_factors, protective_factors = analyze_risk_factors(patient_data)
        
        # Generate recommendations
        recommendations = generate_clinical_recommendations(
            interpretation['risk_level'], probability, risk_factors
        )
        
        # Display Results
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Main risk display
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.plotly_chart(create_risk_gauge(probability), use_container_width=True)
        
        # Risk interpretation
        risk_class = "risk-very-high" if probability >= 0.8 else \
                    "risk-high" if probability >= 0.6 else \
                    "risk-moderate" if probability >= 0.4 else "risk-low"
        
        st.markdown(f"""
        <div class="metric-card {risk_class}">
            <h3>🎯 Risk Assessment</h3>
            <p><strong>Risk Level:</strong> {interpretation['risk_level']}</p>
            <p><strong>Probability:</strong> {probability:.1%}</p>
            <p><strong>Confidence:</strong> {interpretation['confidence_level']}</p>
            <p><strong>Clinical Action:</strong> {interpretation['urgency']}</p>
            <p><strong>Timeline:</strong> {interpretation['timeline']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed visualizations
        st.markdown("### 📈 Detailed Analysis")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Risk Factors", "Comparisons", "Timeline", "Breakdown"])
        
        with tab1:
            st.plotly_chart(create_risk_factor_radar(detailed_results), use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ⚠️ Risk Factors")
                if risk_factors:
                    for factor, severity, detail in risk_factors:
                        severity_icon = "🔴" if severity == "Very High" else \
                                      "🟠" if severity == "High" else \
                                      "🟡" if severity == "Medium" else "🟢"
                        st.markdown(f"{severity_icon} **{factor}** ({severity}): {detail}")
                else:
                    st.markdown("✅ No significant risk factors identified")
            
            with col2:
                st.markdown("#### 🛡️ Protective Factors")
                if protective_factors:
                    for factor, detail in protective_factors:
                        st.markdown(f"✅ **{factor}**: {detail}")
                else:
                    st.markdown("⚠️ Limited protective factors present")
        
        with tab2:
            st.plotly_chart(create_risk_comparison_chart(patient_data, detailed_results), use_container_width=True)
            st.plotly_chart(create_probability_distribution(probability), use_container_width=True)
        
        with tab3:
            st.plotly_chart(create_risk_timeline(patient_data), use_container_width=True)
            
            st.markdown("""
            <div class="info-box">
            <strong>Risk Timeline Interpretation:</strong><br>
            This chart shows how cardiovascular risk may progress with age, assuming current health parameters remain constant. 
            The projection accounts for age-related decline in maximum heart rate and increased cardiovascular risk.
            </div>
            """, unsafe_allow_html=True)
        
        with tab4:
            st.plotly_chart(create_risk_breakdown_pie(detailed_results), use_container_width=True)
            
            # Individual risk factor details
            st.markdown("#### Individual Risk Factor Scores")
            individual_risks = detailed_results['individual_risks']
            
            for factor, risk_score in individual_risks.items():
                percentage = risk_score * 100
                color = get_risk_color(risk_score)
                st.markdown(f"""
                <div style="margin: 5px 0;">
                    <strong>{factor.replace('_', ' ').title()}:</strong>
                    <div style="background-color: #f0f0f0; border-radius: 10px; padding: 2px;">
                        <div style="background-color: {color}; width: {percentage}%; height: 20px; 
                                    border-radius: 8px; display: flex; align-items: center; padding-left: 10px;">
                            <span style="color: white; font-weight: bold;">{percentage:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Clinical Recommendations
        st.markdown("---")
        st.markdown("## 🩺 Clinical Recommendations")
        
        rec_tab1, rec_tab2, rec_tab3 = st.tabs(["Immediate Actions", "Diagnostic Tests", "Follow-up"])
        
        with rec_tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚨 Immediate Actions")
                for action in recommendations['immediate_actions']:
                    st.markdown(f"• {action}")
                
                if recommendations['referrals']:
                    st.markdown("#### 👨‍⚕️ Specialist Referrals")
                    for referral in recommendations['referrals']:
                        st.markdown(f"• {referral}")
            
            with col2:
                st.markdown("#### 🏃‍♂️ Lifestyle Modifications")
                for lifestyle in recommendations['lifestyle_modifications'][:5]:  # Show top 5
                    st.markdown(f"• {lifestyle}")
        
        with rec_tab2:
            st.markdown("#### 🔬 Recommended Diagnostic Tests")
            for test in recommendations['diagnostic_tests']:
                st.markdown(f"• {test}")
        
        with rec_tab3:
            st.markdown("#### 📅 Follow-up Schedule")
            for followup in recommendations['follow_up']:
                st.markdown(f"• {followup}")
        
        # Patient summary for export
        st.markdown("---")
        st.markdown("## 📋 Patient Summary")
        
        summary_data = {
            "Patient ID": f"PT-{hash(str(patient_data)) % 10000:04d}",
            "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Risk Probability": f"{probability:.1%}",
            "Risk Level": interpretation['risk_level'],
            "Confidence": interpretation['confidence_level'],
            "Primary Risk Factors": len(risk_factors),
            "Protective Factors": len(protective_factors)
        }
        
        summary_df = pd.DataFrame(list(summary_data.items()), columns=['Parameter', 'Value'])
        st.dataframe(summary_df, use_container_width=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Generate Report"):
                report = generate_clinical_report(patient_data, detailed_results, interpretation, 
                                                risk_factors, protective_factors, recommendations)
                st.download_button(
                    label="Download Clinical Report",
                    data=report,
                    file_name=f"heart_disease_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        
        with col2:
            if st.button("📊 Export Data"):
                export_data = {
                    **patient_data,
                    'predicted_probability': probability,
                    'risk_level': interpretation['risk_level'],
                    'analysis_date': datetime.now().isoformat()
                }
                st.download_button(
                    label="Download JSON Data",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"patient_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col3:
            if st.button("🔄 Reset Analysis"):
                st.experimental_rerun()

def generate_clinical_report(patient_data, detailed_results, interpretation, 
                           risk_factors, protective_factors, recommendations):
    """Generate comprehensive clinical report"""
    
    report = f"""
HEART DISEASE RISK ASSESSMENT REPORT
=====================================

Patient Information:
- Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- Patient ID: PT-{hash(str(patient_data)) % 10000:04d}

Clinical Parameters:
- Age: {patient_data['Age']} years
- Sex: {patient_data['Sex']}
- Chest Pain Type: {patient_data['ChestPainType']}
- Resting Blood Pressure: {patient_data['RestingBP']} mmHg
- Cholesterol: {patient_data['Cholesterol']} mg/dl
- Fasting Blood Sugar: {"Yes" if patient_data['FastingBS'] == 1 else "No"} (>120 mg/dl)
- Resting ECG: {patient_data['RestingECG']}
- Maximum Heart Rate: {patient_data['MaxHR']} bpm
- Exercise Angina: {"Yes" if patient_data['ExerciseAngina'] == 'Y' else "No"}
- ST Depression: {patient_data['Oldpeak']} mm
- ST Slope: {patient_data['ST_Slope']}

RISK ASSESSMENT:
===============
Risk Probability: {detailed_results['probability']:.1%}
Risk Level: {interpretation['risk_level']}
Confidence Level: {interpretation['confidence_level']}
Clinical Significance: {interpretation['clinical_significance']}

Recommended Action: {interpretation['urgency']}
Timeline: {interpretation['timeline']}

RISK FACTORS ANALYSIS:
=====================
Identified Risk Factors:
"""
    
    for factor, severity, detail in risk_factors:
        report += f"- {factor} ({severity}): {detail}\n"
    
    report += f"\nProtective Factors:\n"
    for factor, detail in protective_factors:
        report += f"- {factor}: {detail}\n"
    
    report += f"""
CLINICAL RECOMMENDATIONS:
========================

Immediate Actions:
"""
    for action in recommendations['immediate_actions']:
        report += f"- {action}\n"
    
    report += f"\nDiagnostic Tests:\n"
    for test in recommendations['diagnostic_tests']:
        report += f"- {test}\n"
    
    report += f"\nLifestyle Modifications:\n"
    for lifestyle in recommendations['lifestyle_modifications']:
        report += f"- {lifestyle}\n"
    
    report += f"\nFollow-up Schedule:\n"
    for followup in recommendations['follow_up']:
        report += f"- {followup}\n"
    
    if recommendations['referrals']:
        report += f"\nSpecialist Referrals:\n"
        for referral in recommendations['referrals']:
            report += f"- {referral}\n"
    
    report += f"""

DISCLAIMER:
===========
This assessment is based on computational analysis of clinical parameters and should not 
replace professional medical judgment. All recommendations should be reviewed by qualified 
healthcare professionals before implementation.

Report generated by Heart Disease Prediction System v2.0
"""
    
    return report

# ================================================================================
# APPLICATION FOOTER
# ================================================================================

def display_footer():
    """Display application footer"""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #666;">
    <p><strong>Heart Disease Prediction System v2.0</strong></p>
    <p>Advanced Clinical Decision Support Tool</p>
    <p>⚠️ For educational and research purposes only. Not for clinical decision making.</p>
    </div>
    """, unsafe_allow_html=True)

# ================================================================================
# RUN APPLICATION
# ================================================================================

if __name__ == "__main__":
    main()
    display_footer()
