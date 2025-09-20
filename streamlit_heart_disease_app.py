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
# SIMPLIFIED CSS STYLING
# ================================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .risk-high { border-left-color: #dc3545; background: #fff5f5; }
    .risk-moderate { border-left-color: #ffc107; background: #fffdf0; }
    .risk-low { border-left-color: #28a745; background: #f8fff8; }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
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
        st.markdown('<h1 class="main-header">❤️ Heart Disease Risk Predictor</h1>', unsafe_allow_html=True)
        
        # Info box
        st.markdown("""
        <div class="info-box">
        <strong>Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice.
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar inputs
        st.sidebar.header("Patient Information")
        
        # Input fields
        age = st.sidebar.slider("Age", 20, 100, 54)
        sex = st.sidebar.selectbox("Sex", ["M", "F"])
        chest_pain_type = st.sidebar.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP", "TA"])
        resting_bp = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 132)
        cholesterol = st.sidebar.slider("Cholesterol (mg/dl)", 100, 400, 246)
        fasting_bs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
        resting_ecg = st.sidebar.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.sidebar.selectbox("Exercise Angina", ["N", "Y"])
        oldpeak = st.sidebar.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, 0.1)
        st_slope = st.sidebar.selectbox("ST Slope", ["Up", "Flat", "Down"])
        
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
        if st.sidebar.button("🔍 Predict Risk", type="primary"):
            
            with st.spinner("Analyzing patient data..."):
                # Initialize predictor
                predictor = HeartDiseasePredictor()
                
                # Make prediction
                results = predictor.predict(patient_data)
                
                if results:
                    probability = results['probability']
                    
                    # Display results
                    st.markdown("## 📊 Analysis Results")
                    
                    # Main metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Risk Probability", f"{probability:.1%}")
                    
                    with col2:
                        risk_level = "High" if probability >= 0.6 else "Moderate" if probability >= 0.3 else "Low"
                        st.metric("Risk Level", risk_level)
                    
                    with col3:
                        st.metric("Confidence", "High" if len(analyze_risk_factors(patient_data)) > 2 else "Moderate")
                    
                    # Risk gauge
                    fig = create_simple_gauge(probability)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk factors analysis
                    risk_factors = analyze_risk_factors(patient_data)
                    
                    if risk_factors:
                        st.markdown("### ⚠️ Identified Risk Factors")
                        for factor, severity, detail in risk_factors:
                            severity_color = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                            st.markdown(f"{severity_color} **{factor}** ({severity}): {detail}")
                    
                    # Risk factors chart
                    if 'individual_risks' in results:
                        chart = create_risk_factors_chart(results['individual_risks'])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                    
                    # Recommendations
                    recommendations = get_recommendations(probability)
                    
                    risk_class = "risk-high" if probability >= 0.6 else "risk-moderate" if probability >= 0.3 else "risk-low"
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>🩺 Clinical Recommendations</h3>
                        <p><strong>Priority:</strong> {recommendations['urgency']}</p>
                        <p><strong>Timeline:</strong> {recommendations['timeline']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Recommended Actions:**")
                    for action in recommendations['actions']:
                        st.markdown(f"• {action}")
                    
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
