# BalzAnalysis: Unified Options & Stock Analysis Suite

## Features
- **Options Analyzer**: Advanced Nifty50 options analytics, dynamic targets/stops, Greeks, confidence scoring, and robust error handling.
- **Stock Analyzer**: Modern UI, summary cards, technical & fundamental metrics, advanced indicators, candlestick pattern detection, risk/reward analytics, SWOT, long-term/short-term/valuation analysis, and color-coded highlights.
- **1-Click Launch**: Double-click `run_app.command` (macOS) for automatic setup and launch.

## Quick Start (macOS)
1. **Clone or Download** this repository.
2. **Navigate** to the `BalzAnalysis` folder in Finder.
3. **Double-click** `run_app.command`.
   - The script will create a virtual environment, install all requirements, and launch the app in your browser.

## Requirements
- macOS with Python 3.8+
- Internet connection (for data)

## Usage
- Use the sidebar to switch between **Options Analyzer** and **Stock Analyzer**.
- Enter a stock symbol (e.g., `RELIANCE`) and click **Analyze** for detailed analytics.
- All analytics are for educational purposes only.

## Project Structure
- `OptionChain/` – Options analytics engine
- `StockAnalysis/` – Stock analytics engine
- `balzanalysis_app.py` – Unified Streamlit app
- `run_app.command` – 1-click launcher (macOS)
- `requirements.txt` – Python dependencies

## Support
For issues or feature requests, please contact the maintainer.

---
**Note:** For Windows, run the following in Command Prompt:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run balzanalysis_app.py
```
