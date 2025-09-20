from pathlib import Path
import os

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = BASE_DIR / "heart_disease_analysis_results" / "deployment"

    BEST_MODEL_PATH = MODEL_DIR / "best_original_joblib.pkl"
    SCALER_PATH = MODEL_DIR / "scaler_joblib.pkl"
    DEPLOYMENT_INFO_JSON = MODEL_DIR / "deployment_info.json"
    DEPLOYMENT_INFO_PKL = MODEL_DIR / "deployment_info.pkl"
    BIAS_RESULTS_PATH = BASE_DIR / "heart_disease_analysis_results" / "bias_generalizability_results.pkl"

    RISK_THRESHOLDS = {
        "low": 0.2,
        "moderate": 0.4,
        "high": 0.6,
        "very_high": 0.8,
    }

    FAIRNESS_THRESHOLDS = {
        "demographic_parity": 0.1,
        "equalized_odds": 0.1,
        "accuracy_difference": 0.05,
    }

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = "heart_disease_app.log"
    CACHE_TTL = 3600
    MAX_PREDICTIONS_PER_HOUR = 100
