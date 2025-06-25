#!/bin/zsh
# 1-Click launcher for BalzAnalysis (macOS)
cd "$(dirname "$0")"

# Create venv if missing
if [ ! -d "../.venv" ]; then
  python3 -m venv ../.venv
fi

# Activate venv
source ../.venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt

# Launch the unified Streamlit app
streamlit run balzanalysis_app.py
