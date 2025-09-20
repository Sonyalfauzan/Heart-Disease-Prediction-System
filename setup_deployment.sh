#!/usr/bin/env bash
set -e

echo "=== Heart Disease AI Deployment Setup ==="

# create venv (optional)
if [ ! -d ".venv" ]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate || true

echo "[1/3] Installing requirements..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "[2/3] Installing optional components..."
pip install streamlit-option-menu streamlit-plotly-events streamlit-aggrid || true

echo "[3/3] Checking deployment files..."
python check_deployment_files.py || true

echo "Setup complete."
echo "Run: streamlit run streamlit_heart_disease_app.py"
