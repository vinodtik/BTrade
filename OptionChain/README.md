# Option Chain Analyzer

This project provides tools to analyze option chain data for the Indian stock market (NIFTY, BANKNIFTY, etc.), generate trade recommendations, and visualize market sentiment. It supports both command-line and Streamlit web app interfaces.

## Features
- **Custom CSV Loader:** Reads option chain CSVs with combined CALL/PUT rows (skips first two header rows).
- **Filtering:** Filters options by liquidity (OI, volume, bid-ask spread) and proximity to spot price.
- **Market Sentiment:** Analyzes OI to determine support, resistance, and sentiment (bullish, bearish, neutral).
- **Trade Recommendations:** Suggests BUY/AVOID actions based on OI/volume analysis, with stop loss and target levels.
- **Reports:** Tabulated output for all recommendations, best scalp, and swing trades.
- **Streamlit App:** Upload CSV, view sentiment, and get recommendations interactively.

## Usage

### 1. Command Line
```sh
python src/main.py <option_chain_csv>
```
Example:
```sh
python src/main.py "option-chain-ED-NIFTY-19-Jun-2025 (2).csv"
```

### 2. Streamlit Web App
```sh
streamlit run src/app.py
```

Upload your option chain CSV file (downloaded from NSE) in the app interface.

## CSV Format
- The loader expects a CSV with combined CALL/PUT rows, skipping the first two header rows (NSE format).
- Required columns: Strike Price, Option Type, LTP, Bid, Ask, OI, Change in OI, Volume, IV, Spot Price.

## Requirements
Install dependencies:
```sh
pip install -r requirements.txt
```

## If you see 'zsh: command not found: streamlit' or have install issues

Your system may restrict direct package installation. Use a virtual environment:

### 1. Create and activate a virtual environment
```sh
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies (including Streamlit)
```sh
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```sh
streamlit run src/app.py
```

If you still have issues, you can install Streamlit directly with:
```sh
pip install streamlit
```

## File Structure
- `src/loader.py` – Loads and parses the option chain CSV
- `src/strategy.py` – Filtering logic
- `src/analyzer.py` – OI/volume analysis
- `src/recommender.py` – Trade recommendation logic
- `src/report.py` – Tabulated report output
- `src/app.py` – Streamlit web app
- `src/main.py` – Command-line entry point
- `src/config.py` – Project constants

## Disclaimer
This tool is for educational and informational purposes only. Not financial advice. Use at your own risk.


## last option 
rm -rf venv && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && streamlit run src/app.py