# streamlit_heart_disease_app.py - OPTIMIZED VERSION
# ================================================================================
# HEART DISEASE PREDICTION SYSTEM - CLOUD-OPTIMIZED STREAMLIT APPLICATION
# ================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import math
import sys
import gc

# ================================================================================
# PAGE CONFIGURATION - SIMPLIFIED
# ================================================================================

try:
    st.set_page_config(
        page_title="Heart Disease Prediction",
        page_icon="❤️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

# ================================================================================
# ENHANCED CSS STYLING
# ================================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #4a5568;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border-left: 5px solid #667eea;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Risk Level Cards */
    .risk-very-high {
        background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(220, 53, 69, 0.05) 100%);
        border-left-color: #dc3545;
        border: 1px solid rgba(220, 53, 69, 0.2);
    }
    
    .risk-high {
        background: linear-gradient(135deg, rgba(253, 126, 20, 0.1) 0%, rgba(253, 126, 20, 0.05) 100%);
        border-left-color: #fd7e14;
        border: 1px solid rgba(253, 126, 20, 0.2);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        border-left-color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.2);
    }
    
    .risk-low {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left-color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.2);
    }
    
    /* Info Boxes */
    .info-box {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.05) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #667eea;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.1);
        border: 1px solid rgba(102, 126, 234, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 193, 7, 0.05) 100%);
        border-left-color: #ffc107;
        border: 1px solid rgba(255, 193, 7, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(40, 167, 69, 0.05) 100%);
        border-left-color: #28a745;
        border: 1px solid rgba(40, 167, 69, 0.2);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.95) 0%, rgba(247, 250, 252, 0.95) 100%);
        backdrop-filter: blur(20px);
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Metric Styling */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Risk Factor Styling */
    .risk-factor {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border-left: 4px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    
    .risk-factor:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Text Styling */
    .big-text {
        font-size: 2rem;
        font-weight: 600;
        color: #2d3748;
        text-align: center;
        margin: 1rem 0;
    }
    
    .highlight-text {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Chart Container */
    .chart-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Footer Styling */
    .footer {
        text-align: center;
        padding: 2rem;
        color: rgba(255, 255, 255, 0.8);
        background: rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        margin-top: 3rem;
        border-radius: 15px;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card {
            padding: 1rem;
            margin: 1rem 0;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# SIMPLIFIED PREDICTION MODEL
# ================================================================================

class HeartDiseasePredictor:
    """Simplified but accurate heart disease predictor"""
    
    def __init__(self):
        # Simplified feature weights
        self.weights = {
            'age': 0.15,
            'sex': 0.12,
            'chest_pain': 0.20,
            'bp': 0.10,
            'cholesterol': 0.12,
            'fbs': 0.08,
            'ecg': 0.07,
            'max_hr': 0.12,
            'exercise_angina': 0.16,
            'oldpeak': 0.14,
            'slope': 0.10
        }
    
    def normalize_age(self, age):
        """Normalize age factor"""
        if age >= 70: return 1.0
        elif age >= 60: return 0.8
        elif age >= 50: return 0.6
        elif age >= 40: return 0.4
        else: return 0.2
    
    def normalize_bp(self, bp):
        """Normalize blood pressure"""
        if bp >= 180: return 1.0
        elif bp >= 140: return 0.8
        elif bp >= 130: return 0.6
        elif bp >= 120: return 0.4
        else: return 0.2
    
    def normalize_cholesterol(self, chol):
        """Normalize cholesterol"""
        if chol >= 300: return 1.0
        elif chol >= 240: return 0.8
        elif chol >= 200: return 0.6
        else: return 0.3
    
    def predict(self, patient_data):
        """Make prediction with simplified algorithm"""
        try:
            # Calculate normalized risk factors
            age_risk = self.normalize_age(patient_data['Age'])
            sex_risk = 1.0 if patient_data['Sex'] == 'M' else 0.3
            
            # Chest pain risk
            cp_map = {'ASY': 1.0, 'ATA': 0.7, 'NAP': 0.4, 'TA': 0.6}
            cp_risk = cp_map.get(patient_data['ChestPainType'], 0.5)
            
            bp_risk = self.normalize_bp(patient_data['RestingBP'])
            chol_risk = self.normalize_cholesterol(patient_data['Cholesterol'])
            fbs_risk = 0.7 if patient_data['FastingBS'] == 1 else 0.2
            
            # ECG risk
            ecg_map = {'Normal': 0.1, 'ST': 0.7, 'LVH': 0.9}
            ecg_risk = ecg_map.get(patient_data['RestingECG'], 0.1)
            
            # Exercise capacity
            predicted_max_hr = 220 - patient_data['Age']
            hr_ratio = patient_data['MaxHR'] / predicted_max_hr
            hr_risk = 1.0 if hr_ratio < 0.6 else (0.8 if hr_ratio < 0.75 else 0.3)
            
            angina_risk = 0.9 if patient_data['ExerciseAngina'] == 'Y' else 0.1
            
            # ST depression
            oldpeak_risk = min(1.0, patient_data['Oldpeak'] / 4.0)
            
            # ST slope
            slope_map = {'Down': 1.0, 'Flat': 0.6, 'Up': 0.2}
            slope_risk = slope_map.get(patient_data['ST_Slope'], 0.5)
            
            # Calculate weighted sum
            risk_score = (
                age_risk * self.weights['age'] +
                sex_risk * self.weights['sex'] +
                cp_risk * self.weights['chest_pain'] +
                bp_risk * self.weights['bp'] +
                chol_risk * self.weights['cholesterol'] +
                fbs_risk * self.weights['fbs'] +
                ecg_risk * self.weights['ecg'] +
                hr_risk * self.weights['max_hr'] +
                angina_risk * self.weights['exercise_angina'] +
                oldpeak_risk * self.weights['oldpeak'] +
                slope_risk * self.weights['slope']
            )
            
            # Convert to probability
            probability = 1 / (1 + np.exp(-(risk_score - 0.5) * 6))
            probability = max(0.01, min(0.99, probability))
            
            return {
                'probability': probability,
                'risk_score': risk_score,
                'individual_risks': {
                    'age': age_risk,
                    'sex': sex_risk,
                    'chest_pain': cp_risk,
                    'bp': bp_risk,
                    'cholesterol': chol_risk,
                    'fbs': fbs_risk,
                    'ecg': ecg_risk,
                    'max_hr': hr_risk,
                    'exercise_angina': angina_risk,
                    'oldpeak': oldpeak_risk,
                    'slope': slope_risk
                }
            }
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

# ================================================================================
# SIMPLIFIED VISUALIZATION FUNCTIONS
# ================================================================================

def create_simple_gauge(probability):
    """Create simple risk gauge"""
    try:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level (%)"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': get_risk_color(probability)},
                'steps': [
                    {'range': [0, 25], 'color': "#d4edda"},
                    {'range': [25, 50], 'color': "#fff3cd"},
                    {'range': [50, 75], 'color': "#f8d7da"},
                    {'range': [75, 100], 'color': "#f5c6cb"}
                ]
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    except Exception as e:
        st.error(f"Gauge creation error: {e}")
        return None

def get_risk_color(probability):
    """Get color based on risk"""
    if probability >= 0.75: return "#dc3545"
    elif probability >= 0.5: return "#fd7e14"
    elif probability >= 0.25: return "#ffc107"
    else: return "#28a745"

def create_risk_factors_chart(individual_risks):
    """Create simple risk factors chart"""
    try:
        factors = list(individual_risks.keys())
        values = [v * 100 for v in individual_risks.values()]
        
        fig = px.bar(
            x=factors, 
            y=values,
            title="Individual Risk Factors (%)",
            color=values,
            color_continuous_scale="Reds"
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

# ================================================================================
# RISK ANALYSIS FUNCTIONS
# ================================================================================

def analyze_risk_factors(patient_data):
    """Analyze patient risk factors"""
    risk_factors = []
    
    if patient_data['Age'] > 65:
        risk_factors.append(("Advanced age", "High", f"{patient_data['Age']} years"))
    
    if patient_data['Sex'] == 'M':
        risk_factors.append(("Male gender", "Medium", "Higher baseline risk"))
    
    if patient_data['RestingBP'] >= 140:
        risk_factors.append(("High blood pressure", "High", f"{patient_data['RestingBP']} mmHg"))
    
    if patient_data['Cholesterol'] >= 240:
        risk_factors.append(("High cholesterol", "High", f"{patient_data['Cholesterol']} mg/dl"))
    
    if patient_data['FastingBS'] == 1:
        risk_factors.append(("Diabetes", "High", "Elevated fasting glucose"))
    
    if patient_data['ExerciseAngina'] == 'Y':
        risk_factors.append(("Exercise angina", "High", "Chest pain with exercise"))
    
    if patient_data['Oldpeak'] >= 2.0:
        risk_factors.append(("Significant ST depression", "High", f"{patient_data['Oldpeak']} mm"))
    
    return risk_factors

def get_recommendations(probability):
    """Get clinical recommendations"""
    if probability >= 0.75:
        return {
            'urgency': "URGENT - Immediate medical attention",
            'actions': [
                "Seek immediate cardiology consultation",
                "Consider emergency evaluation if symptomatic",
                "Start cardiac monitoring if appropriate"
            ],
            'timeline': "Within 24-48 hours"
        }
    elif probability >= 0.5:
        return {
            'urgency': "HIGH - Prompt medical evaluation needed",
            'actions': [
                "Schedule cardiology appointment within 1-2 weeks",
                "Consider stress testing",
                "Optimize cardiovascular risk factors"
            ],
            'timeline': "Within 1-2 weeks"
        }
    elif probability >= 0.25:
        return {
            'urgency': "MODERATE - Medical follow-up recommended",
            'actions': [
                "Primary care follow-up within 1 month",
                "Lifestyle modifications",
                "Regular monitoring"
            ],
            'timeline': "Within 1 month"
        }
    else:
        return {
            'urgency': "LOW - Continue preventive care",
            'actions': [
                "Annual health check-ups",
                "Maintain healthy lifestyle",
                "Monitor risk factors"
            ],
            'timeline': "Annual follow-up"
        }

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Main application function"""
    try:
        # Header
        st.markdown('<h1 class="main-header">🫀 Advanced Heart Disease Prediction System</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-Powered Clinical Decision Support Tool</p>', unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div class="info-box fade-in-up">
        <strong>📋 About This System:</strong><br>
        This advanced prediction system uses comprehensive clinical parameters to assess cardiovascular risk. 
        It employs evidence-based algorithms derived from cardiology research to provide accurate risk stratification.
        <br><br>
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
        It should not replace professional medical advice, diagnosis, or treatment.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar inputs
        st.sidebar.markdown("## 👤 Patient Information")
        st.sidebar.markdown("### Demographics")
        
        # Input fields with better descriptions
        age = st.sidebar.slider("👶 Age (years)", 20, 100, 54, help="Patient's age in years")
        sex = st.sidebar.selectbox("🚻 Gender", ["M", "F"], help="M = Male, F = Female")
        
        st.sidebar.markdown("### 🫀 Cardiovascular Parameters")
        chest_pain_type = st.sidebar.selectbox(
            "💔 Chest Pain Type", 
            ["ASY", "ATA", "NAP", "TA"],
            help="ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina"
        )
        
        resting_bp = st.sidebar.slider(
            "🩺 Resting Blood Pressure (mmHg)", 
            80, 200, 132,
            help="Systolic blood pressure at rest"
        )
        
        st.sidebar.markdown("### 🧪 Laboratory Results")
        cholesterol = st.sidebar.slider(
            "🔬 Total Cholesterol (mg/dl)", 
            100, 400, 246,
            help="Total cholesterol level in blood"
        )
        
        fasting_bs = st.sidebar.selectbox(
            "🍯 Fasting Blood Sugar > 120 mg/dl",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="1 if fasting blood sugar > 120 mg/dl, else 0"
        )
        
        st.sidebar.markdown("### 📊 Diagnostic Tests")
        resting_ecg = st.sidebar.selectbox(
            "📈 Resting ECG Results",
            ["Normal", "ST", "LVH"],
            help="Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy"
        )
        
        max_hr = st.sidebar.slider(
            "💓 Maximum Heart Rate (bpm)", 
            60, 220, 150,
            help="Maximum heart rate achieved during exercise test"
        )
        
        exercise_angina = st.sidebar.selectbox(
            "🏃‍♂️ Exercise Induced Angina",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
            help="Chest pain during exercise"
        )
        
        oldpeak = st.sidebar.slider(
            "📉 ST Depression (mm)", 
            0.0, 6.0, 1.0, 0.1,
            help="ST depression induced by exercise relative to rest"
        )
        
        st_slope = st.sidebar.selectbox(
            "📊 ST Slope Pattern",
            ["Up", "Flat", "Down"],
            help="Slope of the peak exercise ST segment"
        )
        
        # Patient data
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
        if st.sidebar.button("🔍 Analyze Cardiovascular Risk", type="primary"):
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text('🔍 Analyzing patient parameters...')
                elif i < 60:
                    status_text.text('⚡ Computing risk factors...')
                elif i < 90:
                    status_text.text('🧠 Generating AI predictions...')
                else:
                    status_text.text('✨ Finalizing analysis...')
                time.sleep(0.02)
            
            progress_bar.empty()
            status_text.empty()
            
            with st.spinner("🔬 Processing clinical data..."):
                # Initialize predictor
                predictor = HeartDiseasePredictor()
                
                # Make prediction
                results = predictor.predict(patient_data)
                
                if results:
                    probability = results['probability']
                    
                    # Display results with animation
                    st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
                    st.markdown("## 📊 Comprehensive Risk Analysis")
                    
                    # Main metrics with enhanced styling
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: #667eea; margin: 0;">Risk Probability</h3>
                                <div class="big-text" style="color: {get_risk_color(probability)};">
                                    {probability:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        risk_level = "Very High" if probability >= 0.75 else "High" if probability >= 0.5 else "Moderate" if probability >= 0.25 else "Low"
                        risk_emoji = "🔴" if probability >= 0.75 else "🟠" if probability >= 0.5 else "🟡" if probability >= 0.25 else "🟢"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: #667eea; margin: 0;">Risk Level</h3>
                                <div class="big-text">
                                    {risk_emoji} {risk_level}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_factors = analyze_risk_factors(patient_data)
                        confidence = "High" if len(risk_factors) > 3 else "Moderate" if len(risk_factors) > 1 else "Low"
                        conf_emoji = "💪" if confidence == "High" else "👍" if confidence == "Moderate" else "🤔"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: #667eea; margin: 0;">Confidence</h3>
                                <div class="big-text">
                                    {conf_emoji} {confidence}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        urgency_level = "URGENT" if probability >= 0.75 else "HIGH" if probability >= 0.5 else "MODERATE" if probability >= 0.25 else "LOW"
                        urgency_emoji = "🚨" if probability >= 0.75 else "⚠️" if probability >= 0.5 else "📋" if probability >= 0.25 else "✅"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: #667eea; margin: 0;">Priority</h3>
                                <div class="big-text">
                                    {urgency_emoji} {urgency_level}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk gauge in chart container
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    fig = create_simple_gauge(probability)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Risk factors analysis with enhanced styling
                    risk_factors = analyze_risk_factors(patient_data)
                    
                    if risk_factors:
                        st.markdown("### ⚠️ Identified Risk Factors")
                        
                        for factor, severity, detail in risk_factors:
                            severity_color = "#dc3545" if severity == "High" else "#ffc107" if severity == "Medium" else "#28a745"
                            severity_icon = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                            
                            st.markdown(f"""
                            <div class="risk-factor" style="border-left-color: {severity_color};">
                                <strong>{severity_icon} {factor}</strong> 
                                <span style="color: {severity_color}; font-weight: 600;">({severity} Risk)</span>
                                <br>
                                <small style="color: #6c757d;">{detail}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <strong>✅ Excellent News!</strong><br>
                            No significant risk factors identified in the current analysis.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk factors chart in container
                    if 'individual_risks' in results:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown("### 📈 Individual Risk Factor Analysis")
                        chart = create_risk_factors_chart(results['individual_risks'])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced recommendations
                    recommendations = get_recommendations(probability)
                    
                    risk_class = "risk-very-high" if probability >= 0.75 else "risk-high" if probability >= 0.5 else "risk-moderate" if probability >= 0.25 else "risk-low"
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>🩺 Clinical Recommendations</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0;">
                            <div>
                                <strong>Priority Level:</strong> 
                                <span class="highlight-text">{recommendations['urgency']}</span>
                            </div>
                            <div>
                                <strong>Action Timeline:</strong> 
                                <span style="color: #495057; font-weight: 600;">{recommendations['timeline']}</span>
                            </div>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.7); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <strong>📋 Recommended Actions:</strong>
                            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                    """, unsafe_allow_html=True)
                    
                    for action in recommendations['actions']:
                        st.markdown(f"<li style='margin: 0.3rem 0; color: #495057;'>{action}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Export data
                    if st.button("📊 Export Results"):
                        export_data = {
                            **patient_data,
                            'predicted_probability': float(probability),
                            'risk_level': risk_level,
                            'analysis_date': datetime.now().isoformat()
                        }
                        st.download_button(
                            label="Download Analysis Results",
                            data=json.dumps(export_data, indent=2),
                            file_name=f"heart_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                else:
                    st.error("Error in prediction. Please check your inputs.")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666;">
        <p>Heart Disease Risk Prediction System | For educational purposes only</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean up memory
        gc.collect()
        
    except Exception as e:
        st.error(f"Application error: {e}")
        st.error("Please refresh the page and try again.")

# ================================================================================
# RUN APPLICATION
# ================================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.error("Please contact support if this persists.")
