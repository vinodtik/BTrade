# Balram Stock Analyzer Web App

This is a Streamlit-based web application that allows users to analyze Indian stocks by entering the stock name (e.g., "RELIANCE"). The app fetches data from Yahoo Finance, performs technical and fundamental analysis, and provides actionable insights.

# Balram Stock Analyzer

A modern Streamlit web app for deep analysis of Indian stocks, combining both fundamental and technical strategies. The app provides:

- Mix Conclusion: Combined fundamental and technical scoring with clear buy/wait/avoid signals
- Beautiful, modern UI with clear sections for fundamentals, technicals, support/resistance, and recommendations
- Golden crossover and other advanced technical signals
- Support and resistance levels (short-term and 6-month)
- Strong fundamental buy signals highlighted
- "Wait for price" logic if the stock is above intrinsic value

## Features
- Enter any NSE stock name (e.g., RELIANCE, TCS, INFY) and get a detailed, actionable analysis
- All key metrics: PE, ROE, Debt/Equity, Dividend Yield, Promoter/FII holding, growth, volatility, and more
- Technical indicators: RSI, MACD, ADX, CCI, Stochastic, EMAs, Bollinger Bands, Golden Crossover
- Support/resistance levels for both short-term and 6-month windows
- Clear, actionable recommendations: Strong Buy, Wait, or Avoid

## How to Run

1. **Clone or download this repository**
2. **Install Python 3.8+** (recommended: Python 3.10+)
3. **Create and activate a virtual environment:**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install requirements:**
   ```sh
   pip install -r requirements.txt
   ```
5. **Run the app:**
   ```sh
   ./run_app.command
   ```
   Or, if you want to run directly:
   ```sh
   streamlit run stock_analyzer_app.py
   ```

---

If you get a permissions error, run:
```sh
chmod +x run_app.command
```

## Usage
- Enter the NSE stock name (e.g., RELIANCE) in the input box and click "Analyze"
- Review the Mix Conclusion, Fundamental, Technical, and Support/Resistance sections
- Follow the actionable recommendations for your investment decisions

## Notes
- Data is sourced from Yahoo Finance via yfinance. Some metrics may be missing for certain stocks.
- This app is for educational and research purposes only. Always do your own due diligence before investing.

## Author
- Ballu Analysis (2025)
