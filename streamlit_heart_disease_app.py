# streamlit_heart_disease_app.py - ENHANCED VERSION WITH BILINGUAL SUPPORT
# ================================================================================
# HEART DISEASE PREDICTION SYSTEM - CLOUD-OPTIMIZED WITH IMPROVED UI/UX
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
# PAGE CONFIGURATION
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
# TRANSLATIONS DICTIONARY
# ================================================================================

translations = {
    "English": {
        "title": "🫀 Advanced Heart Disease Prediction System",
        "subtitle": "AI-Powered Clinical Decision Support Tool",
        "disclaimer_title": "📋 About This System:",
        "disclaimer_text": "This advanced prediction system uses comprehensive clinical parameters to assess cardiovascular risk. It employs evidence-based algorithms derived from cardiology research to provide accurate risk stratification.",
        "medical_disclaimer": "⚠️ Medical Disclaimer:",
        "medical_disclaimer_text": "This tool is for educational and informational purposes only. It should not replace professional medical advice, diagnosis, or treatment.",
        "language_selector": "🌐 Language / Bahasa",
        "patient_info": "👤 Patient Information",
        "demographics": "### Demographics",
        "age": "👶 Age (years)",
        "age_help": "Patient's age in years. Risk increases after age 45 (men) and 55 (women)",
        "gender": "🚻 Gender",
        "gender_help": "M = Male, F = Female",
        "cardiovascular_params": "### 🫀 Cardiovascular Parameters",
        "chest_pain": "💔 Chest Pain Type",
        "chest_pain_help": "ASY: Asymptomatic, ATA: Atypical Angina, NAP: Non-Anginal Pain, TA: Typical Angina",
        "bp": "🩺 Resting Blood Pressure (mmHg)",
        "bp_help": "Systolic blood pressure at rest",
        "lab_results": "### 🧪 Laboratory Results",
        "cholesterol": "🔬 Total Cholesterol (mg/dl)",
        "cholesterol_help": "Total cholesterol level in blood",
        "fasting_bs": "🍯 Fasting Blood Sugar > 120 mg/dl",
        "fasting_bs_help": "Yes if fasting blood sugar > 120 mg/dl, else No",
        "diagnostic_tests": "### 📊 Diagnostic Tests",
        "ecg": "📈 Resting ECG Results",
        "ecg_help": "Normal: Normal, ST: ST-T wave abnormality, LVH: Left ventricular hypertrophy",
        "max_hr": "💓 Maximum Heart Rate (bpm)",
        "max_hr_help": "Maximum heart rate achieved during exercise test",
        "exercise_angina": "🏃‍♂️ Exercise Induced Angina",
        "exercise_angina_help": "Chest pain during exercise",
        "oldpeak": "📉 ST Depression (mm)",
        "oldpeak_help": "ST depression induced by exercise relative to rest",
        "st_slope": "📊 ST Slope Pattern",
        "st_slope_help": "Slope of the peak exercise ST segment",
        "analyze_button": "🔍 Analyze Cardiovascular Risk",
        "analyzing": "🔬 Processing clinical data...",
        "analyzing_params": "🔍 Analyzing patient parameters...",
        "computing_risks": "⚡ Computing risk factors...",
        "generating_predictions": "🧠 Generating AI predictions...",
        "finalizing": "✨ Finalizing analysis...",
        "results_title": "📊 Comprehensive Risk Analysis",
        "risk_probability": "Risk Probability",
        "risk_level": "Risk Level",
        "confidence": "Confidence",
        "priority": "Priority",
        "identified_risk_factors": "⚠️ Identified Risk Factors",
        "no_risk_factors": "✅ Excellent News!",
        "no_risk_factors_text": "No significant risk factors identified in the current analysis.",
        "individual_analysis": "📈 Individual Risk Factor Analysis",
        "clinical_recommendations": "🩺 Clinical Recommendations",
        "priority_level": "Priority Level:",
        "action_timeline": "Action Timeline:",
        "recommended_actions": "📋 Recommended Actions:",
        "export_results": "📊 Export Results",
        "download_analysis": "Download Analysis Results",
        "footer_text": "Heart Disease Risk Prediction System | For educational purposes only",
        "user_guide_title": "📖 User Guide",
        "guide_how_to": "### How to Use This System:",
        "guide_demographics": "**1. Demographics:**",
        "guide_age": "- **Age**: Enter patient's age in years (20-100)",
        "guide_gender": "- **Gender**: Select M for Male, F for Female",
        "guide_chest_pain": "**2. Chest Pain Types:**",
        "guide_asy": "- **ASY (Asymptomatic)**: No chest pain symptoms",
        "guide_ata": "- **ATA (Atypical Angina)**: Chest pain not typical of heart disease",
        "guide_nap": "- **NAP (Non-Anginal Pain)**: Chest pain unrelated to heart",
        "guide_ta": "- **TA (Typical Angina)**: Classic heart-related chest pain",
        "guide_bp": "**3. Blood Pressure:**",
        "guide_bp_normal": "- Normal: < 120 mmHg",
        "guide_bp_elevated": "- Elevated: 120-129 mmHg",
        "guide_bp_stage1": "- High Stage 1: 130-139 mmHg",
        "guide_bp_stage2": "- High Stage 2: ≥ 140 mmHg",
        "guide_cholesterol": "**4. Cholesterol Levels:**",
        "guide_chol_desirable": "- Desirable: < 200 mg/dl",
        "guide_chol_borderline": "- Borderline: 200-239 mg/dl",
        "guide_chol_high": "- High: ≥ 240 mg/dl",
        "guide_ecg": "**5. ECG Results:**",
        "guide_ecg_normal": "- **Normal**: No abnormalities",
        "guide_ecg_st": "- **ST**: ST-T wave changes",
        "guide_ecg_lvh": "- **LVH**: Left ventricular hypertrophy",
        "bp_very_high": "⚠️ Very high blood pressure - immediate medical attention required",
        "bp_high": "⚠️ High blood pressure - doctor consultation recommended",
        "bp_elevated": "💡 Slightly elevated blood pressure - monitor regularly",
        "bp_normal": "✅ Normal blood pressure",
        "chol_very_high": "⚠️ Very high cholesterol - immediate intervention needed",
        "chol_high": "⚠️ High cholesterol - lifestyle changes recommended",
        "chol_borderline": "💡 Borderline cholesterol - monitor diet",
        "chol_normal": "✅ Optimal cholesterol level",
        "risk_very_high": "Very High",
        "risk_high": "High",
        "risk_moderate": "Moderate",
        "risk_low": "Low",
        "confidence_high": "High",
        "confidence_moderate": "Moderate",
        "confidence_low": "Low",
        "urgent": "URGENT",
        "high_priority": "HIGH",
        "moderate_priority": "MODERATE",
        "low_priority": "LOW"
    },
    "Bahasa Indonesia": {
        "title": "🫀 Sistem Prediksi Penyakit Jantung Lanjutan",
        "subtitle": "Alat Bantu Keputusan Klinis Berbasis AI",
        "disclaimer_title": "📋 Tentang Sistem Ini:",
        "disclaimer_text": "Sistem prediksi canggih ini menggunakan parameter klinis komprehensif untuk menilai risiko kardiovaskular. Sistem ini menggunakan algoritma berbasis bukti yang berasal dari penelitian kardiologi untuk memberikan stratifikasi risiko yang akurat.",
        "medical_disclaimer": "⚠️ Disclaimer Medis:",
        "medical_disclaimer_text": "Alat ini hanya untuk tujuan edukasi dan informasi. Tidak boleh menggantikan saran medis profesional, diagnosis, atau pengobatan.",
        "language_selector": "🌐 Language / Bahasa",
        "patient_info": "👤 Informasi Pasien",
        "demographics": "### Demografi",
        "age": "👶 Usia (tahun)",
        "age_help": "Usia pasien dalam tahun. Risiko meningkat setelah usia 45 (pria) dan 55 (wanita)",
        "gender": "🚻 Jenis Kelamin",
        "gender_help": "L = Laki-laki, P = Perempuan",
        "cardiovascular_params": "### 🫀 Parameter Kardiovaskular",
        "chest_pain": "💔 Jenis Nyeri Dada",
        "chest_pain_help": "ASY: Asimptomatik, ATA: Angina Atipikal, NAP: Nyeri Non-Anginal, TA: Angina Tipikal",
        "bp": "🩺 Tekanan Darah Istirahat (mmHg)",
        "bp_help": "Tekanan darah sistolik saat istirahat",
        "lab_results": "### 🧪 Hasil Laboratorium",
        "cholesterol": "🔬 Kolesterol Total (mg/dl)",
        "cholesterol_help": "Kadar kolesterol total dalam darah",
        "fasting_bs": "🍯 Gula Darah Puasa > 120 mg/dl",
        "fasting_bs_help": "Ya jika gula darah puasa > 120 mg/dl, jika tidak pilih Tidak",
        "diagnostic_tests": "### 📊 Tes Diagnostik",
        "ecg": "📈 Hasil EKG Istirahat",
        "ecg_help": "Normal: Normal, ST: Kelainan gelombang ST-T, LVH: Hipertrofi ventrikel kiri",
        "max_hr": "💓 Denyut Jantung Maksimum (bpm)",
        "max_hr_help": "Denyut jantung maksimum yang dicapai selama tes olahraga",
        "exercise_angina": "🏃‍♂️ Angina Akibat Olahraga",
        "exercise_angina_help": "Nyeri dada saat berolahraga",
        "oldpeak": "📉 Depresi ST (mm)",
        "oldpeak_help": "Depresi ST yang dipicu oleh olahraga relatif terhadap istirahat",
        "st_slope": "📊 Pola Kemiringan ST",
        "st_slope_help": "Kemiringan segmen ST puncak olahraga",
        "analyze_button": "🔍 Analisis Risiko Kardiovaskular",
        "analyzing": "🔬 Memproses data klinis...",
        "analyzing_params": "🔍 Menganalisis parameter pasien...",
        "computing_risks": "⚡ Menghitung faktor risiko...",
        "generating_predictions": "🧠 Menghasilkan prediksi AI...",
        "finalizing": "✨ Menyelesaikan analisis...",
        "results_title": "📊 Analisis Risiko Komprehensif",
        "risk_probability": "Probabilitas Risiko",
        "risk_level": "Tingkat Risiko",
        "confidence": "Kepercayaan",
        "priority": "Prioritas",
        "identified_risk_factors": "⚠️ Faktor Risiko Teridentifikasi",
        "no_risk_factors": "✅ Kabar Baik!",
        "no_risk_factors_text": "Tidak ada faktor risiko signifikan yang teridentifikasi dalam analisis saat ini.",
        "individual_analysis": "📈 Analisis Faktor Risiko Individual",
        "clinical_recommendations": "🩺 Rekomendasi Klinis",
        "priority_level": "Tingkat Prioritas:",
        "action_timeline": "Timeline Tindakan:",
        "recommended_actions": "📋 Tindakan yang Direkomendasikan:",
        "export_results": "📊 Ekspor Hasil",
        "download_analysis": "Unduh Hasil Analisis",
        "footer_text": "Sistem Prediksi Risiko Penyakit Jantung | Hanya untuk tujuan edukasi",
        "user_guide_title": "📖 Panduan Penggunaan",
        "guide_how_to": "### Cara Menggunakan Sistem Ini:",
        "guide_demographics": "**1. Demografi:**",
        "guide_age": "- **Usia**: Masukkan usia pasien dalam tahun (20-100)",
        "guide_gender": "- **Jenis Kelamin**: Pilih L untuk Laki-laki, P untuk Perempuan",
        "guide_chest_pain": "**2. Jenis Nyeri Dada:**",
        "guide_asy": "- **ASY (Asimptomatik)**: Tidak ada gejala nyeri dada",
        "guide_ata": "- **ATA (Angina Atipikal)**: Nyeri dada tidak khas penyakit jantung",
        "guide_nap": "- **NAP (Nyeri Non-Anginal)**: Nyeri dada tidak terkait jantung",
        "guide_ta": "- **TA (Angina Tipikal)**: Nyeri dada klasik terkait jantung",
        "guide_bp": "**3. Tekanan Darah:**",
        "guide_bp_normal": "- Normal: < 120 mmHg",
        "guide_bp_elevated": "- Meningkat: 120-129 mmHg",
        "guide_bp_stage1": "- Tinggi Tahap 1: 130-139 mmHg",
        "guide_bp_stage2": "- Tinggi Tahap 2: ≥ 140 mmHg",
        "guide_cholesterol": "**4. Kadar Kolesterol:**",
        "guide_chol_desirable": "- Diinginkan: < 200 mg/dl",
        "guide_chol_borderline": "- Batas: 200-239 mg/dl",
        "guide_chol_high": "- Tinggi: ≥ 240 mg/dl",
        "guide_ecg": "**5. Hasil EKG:**",
        "guide_ecg_normal": "- **Normal**: Tidak ada kelainan",
        "guide_ecg_st": "- **ST**: Perubahan gelombang ST-T",
        "guide_ecg_lvh": "- **LVH**: Hipertrofi ventrikel kiri",
        "bp_very_high": "⚠️ Tekanan darah sangat tinggi - perlu perhatian medis segera",
        "bp_high": "⚠️ Tekanan darah tinggi - konsultasi dokter disarankan",
        "bp_elevated": "💡 Tekanan darah sedikit meningkat - pantau secara berkala",
        "bp_normal": "✅ Tekanan darah normal",
        "chol_very_high": "⚠️ Kolesterol sangat tinggi - intervensi segera diperlukan",
        "chol_high": "⚠️ Kolesterol tinggi - perubahan gaya hidup direkomendasikan",
        "chol_borderline": "💡 Kolesterol batas - pantau diet",
        "chol_normal": "✅ Kadar kolesterol optimal",
        "risk_very_high": "Sangat Tinggi",
        "risk_high": "Tinggi",
        "risk_moderate": "Sedang",
        "risk_low": "Rendah",
        "confidence_high": "Tinggi",
        "confidence_moderate": "Sedang",
        "confidence_low": "Rendah",
        "urgent": "DARURAT",
        "high_priority": "TINGGI",
        "moderate_priority": "SEDANG",
        "low_priority": "RENDAH"
    }
}

# ================================================================================
# ENHANCED CSS STYLING WITH IMPROVED COLORS
# ================================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --primary-blue: #1e40af;
        --secondary-blue: #3b82f6;
        --success-green: #059669;
        --warning-yellow: #d97706;
        --danger-red: #dc2626;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --bg-light: #f8fafc;
        --border-color: #e5e7eb;
    }
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        color: var(--text-primary);
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--primary-blue);
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 0 2px 4px rgba(30, 64, 175, 0.1);
        animation: fadeInDown 1s ease-out;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: var(--text-secondary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 500;
    }
    
    /* Card Styles with Better Contrast */
    .metric-card {
        background: var(--bg-primary);
        color: var(--text-primary);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid var(--border-color);
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.3s ease;
        border-left: 4px solid var(--primary-blue);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Risk Level Cards with Better Readability */
    .risk-very-high {
        background: #fef2f2;
        border-left-color: var(--danger-red);
        color: #991b1b;
        border: 1px solid #fecaca;
    }
    
    .risk-high {
        background: #fff7ed;
        border-left-color: #ea580c;
        color: #9a3412;
        border: 1px solid #fed7aa;
    }
    
    .risk-moderate {
        background: #fffbeb;
        border-left-color: var(--warning-yellow);
        color: #92400e;
        border: 1px solid #fde68a;
    }
    
    .risk-low {
        background: #f0fdf4;
        border-left-color: var(--success-green);
        color: #166534;
        border: 1px solid #bbf7d0;
    }
    
    /* Info Boxes */
    .info-box {
        background: #eff6ff;
        color: var(--text-primary);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--secondary-blue);
        margin: 1.5rem 0;
        border: 1px solid #dbeafe;
    }
    
    .warning-box {
        background: #fffbeb;
        color: #92400e;
        border-left-color: var(--warning-yellow);
        border: 1px solid #fde68a;
    }
    
    .success-box {
        background: #f0fdf4;
        color: #166534;
        border-left-color: var(--success-green);
        border: 1px solid #bbf7d0;
    }
    
    /* Enhanced Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(30, 64, 175, 0.5);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 15px -3px rgba(30, 64, 175, 0.5);
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
    }
    
    /* Metric Container Styling */
    [data-testid="metric-container"] {
        background: var(--bg-primary);
        border: 1px solid var(--border-color);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Risk Factor Cards */
    .risk-factor {
        background: var(--bg-primary);
        color: var(--text-primary);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid transparent;
        transition: all 0.3s ease;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid var(--border-color);
    }
    
    .risk-factor:hover {
        transform: translateX(3px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Chart Container */
    .chart-container {
        background: var(--bg-primary);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-color);
        box-shadow: 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Text Styling */
    .big-text {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text-primary);
        text-align: center;
        margin: 1rem 0;
    }
    
    .highlight-text {
        color: var(--primary-blue);
        font-weight: 600;
    }
    
    /* Animations */
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(135deg, var(--primary-blue) 0%, var(--secondary-blue) 100%);
    }
    
    /* Sidebar Enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-light) 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: var(--text-secondary);
        background: var(--bg-light);
        margin-top: 3rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
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
        
        .big-text {
            font-size: 1.5rem;
        }
        
        .stButton > button {
            width: 100%;
            margin: 0.5rem 0;
        }
    }
</style>
""", unsafe_allow_html=True)

# ================================================================================
# PREDICTION MODEL (SIMPLIFIED BUT ACCURATE)
# ================================================================================

class HeartDiseasePredictor:
    """Enhanced heart disease predictor with bilingual support"""
    
    def __init__(self):
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
        if age >= 70: return 1.0
        elif age >= 60: return 0.8
        elif age >= 50: return 0.6
        elif age >= 40: return 0.4
        else: return 0.2
    
    def normalize_bp(self, bp):
        if bp >= 180: return 1.0
        elif bp >= 140: return 0.8
        elif bp >= 130: return 0.6
        elif bp >= 120: return 0.4
        else: return 0.2
    
    def normalize_cholesterol(self, chol):
        if chol >= 300: return 1.0
        elif chol >= 240: return 0.8
        elif chol >= 200: return 0.6
        else: return 0.3
    
    def predict(self, patient_data):
        try:
            age_risk = self.normalize_age(patient_data['Age'])
            sex_risk = 1.0 if patient_data['Sex'] in ['M', 'L'] else 0.3
            
            cp_map = {'ASY': 1.0, 'ATA': 0.7, 'NAP': 0.4, 'TA': 0.6}
            cp_risk = cp_map.get(patient_data['ChestPainType'], 0.5)
            
            bp_risk = self.normalize_bp(patient_data['RestingBP'])
            chol_risk = self.normalize_cholesterol(patient_data['Cholesterol'])
            fbs_risk = 0.7 if patient_data['FastingBS'] == 1 else 0.2
            
            ecg_map = {'Normal': 0.1, 'ST': 0.7, 'LVH': 0.9}
            ecg_risk = ecg_map.get(patient_data['RestingECG'], 0.1)
            
            predicted_max_hr = 220 - patient_data['Age']
            hr_ratio = patient_data['MaxHR'] / predicted_max_hr
            hr_risk = 1.0 if hr_ratio < 0.6 else (0.8 if hr_ratio < 0.75 else 0.3)
            
            angina_risk = 0.9 if patient_data['ExerciseAngina'] in ['Y', 'Ya'] else 0.1
            oldpeak_risk = min(1.0, patient_data['Oldpeak'] / 4.0)
            
            slope_map = {'Down': 1.0, 'Flat': 0.6, 'Up': 0.2}
            slope_risk = slope_map.get(patient_data['ST_Slope'], 0.5)
            
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
# VISUALIZATION FUNCTIONS
# ================================================================================

def create_simple_gauge(probability):
    """Create enhanced risk gauge with better colors"""
    try:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risk Level (%)", 'font': {'size': 20, 'color': '#1f2937'}},
            number = {'font': {'size': 36, 'color': get_risk_color(probability)}},
            gauge = {
                'axis': {'range': [None, 100], 'tickcolor': "#6b7280", 'tickfont': {'size': 12}},
                'bar': {'color': get_risk_color(probability), 'thickness': 0.75},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#e5e7eb",
                'steps': [
                    {'range': [0, 25], 'color': '#f0fdf4'},
                    {'range': [25, 50], 'color': '#fffbeb'},
                    {'range': [50, 75], 'color': '#fff7ed'},
                    {'range': [75, 100], 'color': '#fef2f2'}
                ],
                'threshold': {
                    'line': {'color': "#dc2626", 'width': 3},
                    'thickness': 0.75,
                    'value': 85
                }
            }
        ))
        
        fig.update_layout(
            height=350,
            font={'color': "#1f2937", 'family': "Inter"},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    except Exception as e:
        st.error(f"Gauge creation error: {e}")
        return None

def get_risk_color(probability):
    """Get color based on risk with improved accessibility"""
    if probability >= 0.75: return "#dc2626"
    elif probability >= 0.5: return "#ea580c"
    elif probability >= 0.25: return "#d97706"
    else: return "#059669"

def create_risk_factors_chart(individual_risks):
    """Create enhanced risk factors chart"""
    try:
        factors = [factor.replace('_', ' ').title() for factor in individual_risks.keys()]
        values = [v * 100 for v in individual_risks.values()]
        
        # Create color scale based on values
        colors = [get_risk_color(v/100) for v in values]
        
        fig = px.bar(
            x=factors, 
            y=values,
            title="Individual Risk Factors (%)",
            color=values,
            color_continuous_scale=[[0, "#059669"], [0.25, "#d97706"], [0.5, "#ea580c"], [1, "#dc2626"]],
            labels={'x': 'Risk Factors', 'y': 'Risk Score (%)'}
        )
        
        fig.update_layout(
            height=450,
            xaxis_tickangle=-45,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font={'color': '#1f2937', 'family': 'Inter'},
            title_font_size=16,
            title_x=0.5
        )
        
        fig.update_traces(
            texttemplate='%{y:.1f}%',
            textposition='outside',
            textfont_size=10
        )
        
        return fig
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

# ================================================================================
# ANALYSIS FUNCTIONS WITH BILINGUAL SUPPORT
# ================================================================================

def analyze_risk_factors(patient_data, language="English"):
    """Analyze patient risk factors with bilingual support"""
    risk_factors = []
    
    # Age analysis
    if patient_data['Age'] > 65:
        if language == "English":
            risk_factors.append(("Advanced age", "High", f"{patient_data['Age']} years"))
        else:
            risk_factors.append(("Usia lanjut", "Tinggi", f"{patient_data['Age']} tahun"))
    
    # Gender analysis
    if patient_data['Sex'] in ['M', 'L']:
        if language == "English":
            risk_factors.append(("Male gender", "Medium", "Higher baseline cardiovascular risk"))
        else:
            risk_factors.append(("Jenis kelamin laki-laki", "Sedang", "Risiko kardiovaskular dasar lebih tinggi"))
    
    # Blood pressure analysis
    if patient_data['RestingBP'] >= 140:
        if language == "English":
            risk_factors.append(("High blood pressure", "High", f"{patient_data['RestingBP']} mmHg"))
        else:
            risk_factors.append(("Tekanan darah tinggi", "Tinggi", f"{patient_data['RestingBP']} mmHg"))
    
    # Cholesterol analysis
    if patient_data['Cholesterol'] >= 240:
        if language == "English":
            risk_factors.append(("High cholesterol", "High", f"{patient_data['Cholesterol']} mg/dl"))
        else:
            risk_factors.append(("Kolesterol tinggi", "Tinggi", f"{patient_data['Cholesterol']} mg/dl"))
    
    # Diabetes analysis
    if patient_data['FastingBS'] == 1:
        if language == "English":
            risk_factors.append(("Diabetes", "High", "Elevated fasting glucose"))
        else:
            risk_factors.append(("Diabetes", "Tinggi", "Glukosa puasa tinggi"))
    
    # Exercise angina analysis
    if patient_data['ExerciseAngina'] in ['Y', 'Ya']:
        if language == "English":
            risk_factors.append(("Exercise angina", "High", "Chest pain with exercise"))
        else:
            risk_factors.append(("Angina saat olahraga", "Tinggi", "Nyeri dada saat berolahraga"))
    
    # ST depression analysis
    if patient_data['Oldpeak'] >= 2.0:
        if language == "English":
            risk_factors.append(("Significant ST depression", "High", f"{patient_data['Oldpeak']} mm"))
        else:
            risk_factors.append(("Depresi ST signifikan", "Tinggi", f"{patient_data['Oldpeak']} mm"))
    
    return risk_factors

def get_recommendations(probability, language="English"):
    """Get clinical recommendations with bilingual support"""
    if language == "English":
        if probability >= 0.75:
            return {
                'urgency': "URGENT - Immediate medical attention required",
                'actions': [
                    "Seek immediate cardiology consultation",
                    "Consider emergency evaluation if symptomatic",
                    "Start cardiac monitoring if appropriate",
                    "Initiate dual antiplatelet therapy if indicated"
                ],
                'timeline': "Within 24-48 hours"
            }
        elif probability >= 0.5:
            return {
                'urgency': "HIGH - Prompt medical evaluation needed",
                'actions': [
                    "Schedule cardiology appointment within 1-2 weeks",
                    "Consider stress testing or imaging",
                    "Optimize cardiovascular risk factors",
                    "Start or optimize statin therapy"
                ],
                'timeline': "Within 1-2 weeks"
            }
        elif probability >= 0.25:
            return {
                'urgency': "MODERATE - Medical follow-up recommended",
                'actions': [
                    "Primary care follow-up within 1 month",
                    "Implement lifestyle modifications",
                    "Regular blood pressure and cholesterol monitoring",
                    "Consider cardiac risk assessment tools"
                ],
                'timeline': "Within 1 month"
            }
        else:
            return {
                'urgency': "LOW - Continue preventive care",
                'actions': [
                    "Annual comprehensive health check-ups",
                    "Maintain heart-healthy lifestyle",
                    "Monitor and control risk factors",
                    "Regular exercise and healthy diet"
                ],
                'timeline': "Annual follow-up"
            }
    else:  # Bahasa Indonesia
        if probability >= 0.75:
            return {
                'urgency': "DARURAT - Perlu perhatian medis segera",
                'actions': [
                    "Segera konsultasi dengan ahli jantung",
                    "Pertimbangkan evaluasi darurat jika bergejala",
                    "Mulai pemantauan jantung jika sesuai",
                    "Inisiasi terapi antiplatelet ganda jika diindikasikan"
                ],
                'timeline': "Dalam 24-48 jam"
            }
        elif probability >= 0.5:
            return {
                'urgency': "TINGGI - Evaluasi medis segera diperlukan",
                'actions': [
                    "Jadwalkan konsultasi kardiologi dalam 1-2 minggu",
                    "Pertimbangkan tes stress atau pencitraan",
                    "Optimalkan faktor risiko kardiovaskular",
                    "Mulai atau optimalkan terapi statin"
                ],
                'timeline': "Dalam 1-2 minggu"
            }
        elif probability >= 0.25:
            return {
                'urgency': "SEDANG - Tindak lanjut medis direkomendasikan",
                'actions': [
                    "Tindak lanjut perawatan primer dalam 1 bulan",
                    "Implementasikan modifikasi gaya hidup",
                    "Pemantauan rutin tekanan darah dan kolesterol",
                    "Pertimbangkan alat penilaian risiko jantung"
                ],
                'timeline': "Dalam 1 bulan"
            }
        else:
            return {
                'urgency': "RENDAH - Lanjutkan perawatan preventif",
                'actions': [
                    "Pemeriksaan kesehatan komprehensif tahunan",
                    "Pertahankan gaya hidup sehat jantung",
                    "Pantau dan kontrol faktor risiko",
                    "Olahraga rutin dan diet sehat"
                ],
                'timeline': "Tindak lanjut tahunan"
            }

# ================================================================================
# INPUT VALIDATION FUNCTIONS
# ================================================================================

def validate_bp(bp_value, language="English"):
    """Validate blood pressure and provide feedback"""
    if language == "English":
        if bp_value >= 180:
            return "bp_very_high"
        elif bp_value >= 140:
            return "bp_high"
        elif bp_value >= 120:
            return "bp_elevated"
        else:
            return "bp_normal"
    else:
        if bp_value >= 180:
            return "bp_very_high"
        elif bp_value >= 140:
            return "bp_high"
        elif bp_value >= 120:
            return "bp_elevated"
        else:
            return "bp_normal"

def validate_cholesterol(chol_value, language="English"):
    """Validate cholesterol and provide feedback"""
    if language == "English":
        if chol_value >= 300:
            return "chol_very_high"
        elif chol_value >= 240:
            return "chol_high"
        elif chol_value >= 200:
            return "chol_borderline"
        else:
            return "chol_normal"
    else:
        if chol_value >= 300:
            return "chol_very_high"
        elif chol_value >= 240:
            return "chol_high"
        elif chol_value >= 200:
            return "chol_borderline"
        else:
            return "chol_normal"

# ================================================================================
# MAIN APPLICATION
# ================================================================================

def main():
    """Enhanced main application with bilingual support"""
    try:
        # Language selection
        language = st.sidebar.selectbox(
            "🌐 Language / Bahasa", 
            ["English", "Bahasa Indonesia"],
            help="Select your preferred language / Pilih bahasa yang diinginkan"
        )
        
        # Get translations
        t = translations[language]
        
        # Header
        st.markdown(f'<h1 class="main-header">{t["title"]}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p class="sub-header">{t["subtitle"]}</p>', unsafe_allow_html=True)
        
        # User Guide
        with st.expander(t["user_guide_title"]):
            st.markdown(f"""
            {t["guide_how_to"]}
            
            {t["guide_demographics"]}
            {t["guide_age"]}
            {t["guide_gender"]}
            
            {t["guide_chest_pain"]}
            {t["guide_asy"]}
            {t["guide_ata"]}
            {t["guide_nap"]}
            {t["guide_ta"]}
            
            {t["guide_bp"]}
            {t["guide_bp_normal"]}
            {t["guide_bp_elevated"]}
            {t["guide_bp_stage1"]}
            {t["guide_bp_stage2"]}
            
            {t["guide_cholesterol"]}
            {t["guide_chol_desirable"]}
            {t["guide_chol_borderline"]}
            {t["guide_chol_high"]}
            
            {t["guide_ecg"]}
            {t["guide_ecg_normal"]}
            {t["guide_ecg_st"]}
            {t["guide_ecg_lvh"]}
            """)
        
        # Info box
        st.markdown(f"""
        <div class="info-box fade-in-up">
        <strong>{t["disclaimer_title"]}</strong><br>
        {t["disclaimer_text"]}
        <br><br>
        <strong>{t["medical_disclaimer"]}</strong> {t["medical_disclaimer_text"]}
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar inputs
        st.sidebar.markdown(f"## {t['patient_info']}")
        st.sidebar.markdown(t["demographics"])
        
        # Input fields with enhanced validation
        age = st.sidebar.slider(
            t["age"], 
            20, 100, 54,
            help=t["age_help"]
        )
        
        if language == "English":
            sex_options = ["M", "F"]
            sex_labels = ["Male", "Female"]
        else:
            sex_options = ["L", "P"]
            sex_labels = ["Laki-laki", "Perempuan"]
        
        sex = st.sidebar.selectbox(
            t["gender"], 
            sex_options,
            format_func=lambda x: sex_labels[sex_options.index(x)],
            help=t["gender_help"]
        )
        
        st.sidebar.markdown(t["cardiovascular_params"])
        chest_pain_type = st.sidebar.selectbox(
            t["chest_pain"],
            ["ASY", "ATA", "NAP", "TA"],
            help=t["chest_pain_help"]
        )
        
        resting_bp = st.sidebar.slider(
            t["bp"],
            80, 200, 132,
            help=t["bp_help"]
        )
        
        # Blood pressure validation feedback
        bp_status = validate_bp(resting_bp, language)
        if bp_status == "bp_very_high":
            st.sidebar.error(t[bp_status])
        elif bp_status == "bp_high":
            st.sidebar.warning(t[bp_status])
        elif bp_status == "bp_elevated":
            st.sidebar.info(t[bp_status])
        else:
            st.sidebar.success(t[bp_status])
        
        st.sidebar.markdown(t["lab_results"])
        cholesterol = st.sidebar.slider(
            t["cholesterol"],
            100, 400, 246,
            help=t["cholesterol_help"]
        )
        
        # Cholesterol validation feedback
        chol_status = validate_cholesterol(cholesterol, language)
        if chol_status == "chol_very_high":
            st.sidebar.error(t[chol_status])
        elif chol_status == "chol_high":
            st.sidebar.warning(t[chol_status])
        elif chol_status == "chol_borderline":
            st.sidebar.info(t[chol_status])
        else:
            st.sidebar.success(t[chol_status])
        
        fasting_bs = st.sidebar.selectbox(
            t["fasting_bs"],
            [0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak" if language == "Bahasa Indonesia" else ("Yes" if x == 1 else "No"),
            help=t["fasting_bs_help"]
        )
        
        st.sidebar.markdown(t["diagnostic_tests"])
        resting_ecg = st.sidebar.selectbox(
            t["ecg"],
            ["Normal", "ST", "LVH"],
            help=t["ecg_help"]
        )
        
        max_hr = st.sidebar.slider(
            t["max_hr"],
            60, 220, 150,
            help=t["max_hr_help"]
        )
        
        exercise_angina = st.sidebar.selectbox(
            t["exercise_angina"],
            ["N", "Y"] if language == "English" else ["Tidak", "Ya"],
            format_func=lambda x: "Yes" if x == "Y" else "No" if language == "English" else x,
            help=t["exercise_angina_help"]
        )
        
        oldpeak = st.sidebar.slider(
            t["oldpeak"],
            0.0, 6.0, 1.0, 0.1,
            help=t["oldpeak_help"]
        )
        
        st_slope = st.sidebar.selectbox(
            t["st_slope"],
            ["Up", "Flat", "Down"],
            help=t["st_slope_help"]
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
            'ExerciseAngina': exercise_angina if language == "English" else ("Y" if exercise_angina == "Ya" else "N"),
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }
        
        # Prediction button
        if st.sidebar.button(t["analyze_button"], type="primary"):
            
            # Progress indicator
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i in range(100):
                progress_bar.progress(i + 1)
                if i < 30:
                    status_text.text(t["analyzing_params"])
                elif i < 60:
                    status_text.text(t["computing_risks"])
                elif i < 90:
                    status_text.text(t["generating_predictions"])
                else:
                    status_text.text(t["finalizing"])
                time.sleep(0.02)
            
            progress_bar.empty()
            status_text.empty()
            
            with st.spinner(t["analyzing"]):
                # Initialize predictor
                predictor = HeartDiseasePredictor()
                
                # Make prediction
                results = predictor.predict(patient_data)
                
                if results:
                    probability = results['probability']
                    
                    # Display results with enhanced styling
                    st.markdown('<div class="fade-in-up">', unsafe_allow_html=True)
                    st.markdown(f"## {t['results_title']}")
                    
                    # Main metrics with enhanced styling
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: var(--primary-blue); margin: 0; font-size: 1.1rem;">{t["risk_probability"]}</h3>
                                <div class="big-text" style="color: {get_risk_color(probability)};">
                                    {probability:.1%}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if probability >= 0.75:
                            risk_level = t["risk_very_high"]
                            risk_emoji = "🔴"
                        elif probability >= 0.5:
                            risk_level = t["risk_high"]
                            risk_emoji = "🟠"
                        elif probability >= 0.25:
                            risk_level = t["risk_moderate"]
                            risk_emoji = "🟡"
                        else:
                            risk_level = t["risk_low"]
                            risk_emoji = "🟢"
                            
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: var(--primary-blue); margin: 0; font-size: 1.1rem;">{t["risk_level"]}</h3>
                                <div class="big-text">
                                    {risk_emoji} {risk_level}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        risk_factors = analyze_risk_factors(patient_data, language)
                        if len(risk_factors) > 3:
                            confidence = t["confidence_high"]
                            conf_emoji = "💪"
                        elif len(risk_factors) > 1:
                            confidence = t["confidence_moderate"]
                            conf_emoji = "👍"
                        else:
                            confidence = t["confidence_low"]
                            conf_emoji = "🤔"
                            
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: var(--primary-blue); margin: 0; font-size: 1.1rem;">{t["confidence"]}</h3>
                                <div class="big-text">
                                    {conf_emoji} {confidence}
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        if probability >= 0.75:
                            urgency_level = t["urgent"]
                            urgency_emoji = "🚨"
                        elif probability >= 0.5:
                            urgency_level = t["high_priority"]
                            urgency_emoji = "⚠️"
                        elif probability >= 0.25:
                            urgency_level = t["moderate_priority"]
                            urgency_emoji = "📋"
                        else:
                            urgency_level = t["low_priority"]
                            urgency_emoji = "✅"
                            
                        st.markdown(f"""
                        <div class="metric-card">
                            <div style="text-align: center;">
                                <h3 style="color: var(--primary-blue); margin: 0; font-size: 1.1rem;">{t["priority"]}</h3>
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
                    if risk_factors:
                        st.markdown(f"### {t['identified_risk_factors']}")
                        
                        for factor, severity, detail in risk_factors:
                            if language == "English":
                                severity_color = "#dc2626" if severity == "High" else "#d97706" if severity == "Medium" else "#059669"
                                severity_icon = "🔴" if severity == "High" else "🟡" if severity == "Medium" else "🟢"
                                severity_text = severity
                            else:
                                severity_color = "#dc2626" if severity == "Tinggi" else "#d97706" if severity == "Sedang" else "#059669"
                                severity_icon = "🔴" if severity == "Tinggi" else "🟡" if severity == "Sedang" else "🟢"
                                severity_text = severity
                            
                            st.markdown(f"""
                            <div class="risk-factor" style="border-left-color: {severity_color};">
                                <strong>{severity_icon} {factor}</strong> 
                                <span style="color: {severity_color}; font-weight: 600;">({severity_text} Risk)</span>
                                <br>
                                <small style="color: #6b7280;">{detail}</small>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <strong>{t["no_risk_factors"]}</strong><br>
                            {t["no_risk_factors_text"]}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Risk factors chart in container
                    if 'individual_risks' in results:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown(f"### {t['individual_analysis']}")
                        chart = create_risk_factors_chart(results['individual_risks'])
                        if chart:
                            st.plotly_chart(chart, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Enhanced recommendations
                    recommendations = get_recommendations(probability, language)
                    
                    risk_class = "risk-very-high" if probability >= 0.75 else "risk-high" if probability >= 0.5 else "risk-moderate" if probability >= 0.25 else "risk-low"
                    
                    st.markdown(f"""
                    <div class="metric-card {risk_class}">
                        <h3>{t["clinical_recommendations"]}</h3>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin: 1rem 0; flex-wrap: wrap;">
                            <div style="margin-bottom: 0.5rem;">
                                <strong>{t["priority_level"]}</strong> 
                                <span class="highlight-text">{recommendations['urgency']}</span>
                            </div>
                            <div style="margin-bottom: 0.5rem;">
                                <strong>{t["action_timeline"]}</strong> 
                                <span style="color: #4b5563; font-weight: 600;">{recommendations['timeline']}</span>
                            </div>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <strong>{t["recommended_actions"]}</strong>
                            <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
                    """, unsafe_allow_html=True)
                    
                    for action in recommendations['actions']:
                        st.markdown(f"<li style='margin: 0.4rem 0; color: #374151;'>{action}</li>", unsafe_allow_html=True)
                    
                    st.markdown("""
                            </ul>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Export functionality
                    if st.button(t["export_results"]):
                        export_data = {
                            **patient_data,
                            'predicted_probability': float(probability),
                            'risk_level': risk_level,
                            'confidence': confidence,
                            'recommendations': recommendations,
                            'risk_factors': risk_factors,
                            'analysis_date': datetime.now().isoformat(),
                            'language': language
                        }
                        st.download_button(
                            label=t["download_analysis"],
                            data=json.dumps(export_data, indent=2, ensure_ascii=False),
                            file_name=f"heart_risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                else:
                    st.error("Error in prediction. Please check your inputs." if language == "English" 
                           else "Error dalam prediksi. Mohon periksa input Anda.")
        
        # Footer
        st.markdown(f"""
        <div class="footer">
            <p><strong>{t["footer_text"]}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clean up memory
        gc.collect()
        
    except Exception as e:
        error_msg = f"Application error: {e}" if language == "English" else f"Error aplikasi: {e}"
        st.error(error_msg)
        refresh_msg = "Please refresh the page and try again." if language == "English" else "Mohon refresh halaman dan coba lagi."
        st.error(refresh_msg)

# ================================================================================
# RUN APPLICATION
# ================================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Fatal error: {e}")
        st.error("Please contact support if this persists." if st.session_state.get('language', 'English') == 'English' 
               else "Mohon hubungi dukungan jika masalah berlanjut.")
