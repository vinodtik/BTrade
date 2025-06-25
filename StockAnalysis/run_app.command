#!/bin/zsh
# One-click launcher for the Balram Stock Analyzer Streamlit app

# Ensure the script exits on error
set -e

# Activate the virtual environment
source "$(dirname "$0")/venv/bin/activate"

# Run the Streamlit app
streamlit run stock_analyzer_app.py
