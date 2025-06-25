import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta
import ta  # For technical indicators
from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import StochasticOscillator
from ta.volatility import BollingerBands
import pandas_ta as pta
import mplfinance as mpf
import uuid

# ----------------------
# Utility Functions
# ----------------------
# Add Streamlit caching for expensive operations
@st.cache_data(show_spinner=False)
def fetch_stock_data(stock_name):
    ticker = f"{stock_name.upper()}.NS"
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        if hist.empty or not info:
            raise ValueError("No data found.")
        # Calculate additional metrics
        hist['Returns'] = hist['Close'].pct_change()
        volatility = hist['Returns'].std() * (252 ** 0.5)  # Annualized
        info['volatility'] = volatility
        return hist, info  # Only return serializable objects
    except Exception as e:
        return None, None

@st.cache_data(show_spinner=False)
def calculate_technical_indicators(hist):
    result = {}
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    # MACD
    macd = MACD(close)
    result['MACD'] = macd.macd().iloc[-1]
    result['MACD_signal'] = macd.macd_signal().iloc[-1]
    # CCI
    cci = CCIIndicator(close=close, high=high, low=low)
    result['CCI'] = cci.cci().iloc[-1]
    # ADX
    adx = ADXIndicator(high=high, low=low, close=close)
    result['ADX'] = adx.adx().iloc[-1]
    # Stochastic Oscillator
    stoch = StochasticOscillator(close=close, high=high, low=low)
    result['Stoch_K'] = stoch.stoch().iloc[-1]
    result['Stoch_D'] = stoch.stoch_signal().iloc[-1]
    # Bollinger Bands
    bb = BollingerBands(close)
    result['BB_High'] = bb.bollinger_hband().iloc[-1]
    result['BB_Low'] = bb.bollinger_lband().iloc[-1]
    result['BB_Width'] = bb.bollinger_wband().iloc[-1]
    return result

def fetch_additional_fundamentals(info):
    # PEG Ratio
    pe = info.get('trailingPE', None)
    eps_growth = info.get('earningsQuarterlyGrowth', None)
    peg = pe / (eps_growth * 100) if pe and eps_growth else None
    # Current & Quick Ratio
    current_ratio = info.get('currentRatio', None)
    quick_ratio = info.get('quickRatio', None)
    # Margins
    op_margin = info.get('operatingMargins', None)
    net_margin = info.get('netMargins', None)
    # ROCE
    roce = info.get('returnOnCapitalEmployed', None)
    # Earnings Growth (3Y/5Y CAGR) - fallback to revenueGrowth if not available
    earnings_growth_3y = info.get('earningsGrowth', None)
    earnings_growth_5y = info.get('revenueGrowth', None)
    return {
        'PEG': peg,
        'Current Ratio': current_ratio,
        'Quick Ratio': quick_ratio,
        'Operating Margin': op_margin,
        'Net Profit Margin': net_margin,
        'ROCE': roce,
        'Earnings Growth 3Y': earnings_growth_3y,
        'Earnings Growth 5Y': earnings_growth_5y
    }

def calculate_advanced_technicals(hist):
    result = {}
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    # ATR
    result['ATR'] = ta.volatility.AverageTrueRange(high, low, close).average_true_range().iloc[-1]
    # Parabolic SAR
    result['SAR'] = ta.trend.PSARIndicator(high, low, close).psar().iloc[-1]
    # Supertrend (custom, as ta does not have it natively)
    try:
        atr = ta.volatility.AverageTrueRange(high, low, close).average_true_range()
        hl2 = (high + low) / 2
        factor = 3
        supertrend = hl2 - (factor * atr)
        result['Supertrend'] = supertrend.iloc[-1]
    except Exception:
        result['Supertrend'] = None
    # 200-day SMA/EMA
    result['SMA200'] = close.rolling(window=200).mean().iloc[-1]
    result['EMA200'] = close.ewm(span=200).mean().iloc[-1]
    # Volume analysis
    result['Volume'] = hist['Volume'].iloc[-1]
    result['AvgVolume20'] = hist['Volume'].rolling(window=20).mean().iloc[-1]
    result['AvgVolume50'] = hist['Volume'].rolling(window=50).mean().iloc[-1]
    # Ichimoku Cloud
    ichimoku = pta.ichimoku(high, low, close)
    result['Ichimoku_A'] = ichimoku[0].iloc[-1]
    result['Ichimoku_B'] = ichimoku[1].iloc[-1]
    # Pivot Points
    last = hist.iloc[-1]
    pivot = (last['High'] + last['Low'] + last['Close']) / 3
    result['Pivot'] = pivot
    result['S1'] = (2 * pivot) - last['High']
    result['S2'] = pivot - (last['High'] - last['Low'])
    result['R1'] = (2 * pivot) - last['Low']
    result['R2'] = pivot + (last['High'] - last['Low'])
    return result

def detect_candlestick_patterns(hist):
    # Simple pattern detection for last candle
    open_ = hist['Open'].iloc[-2:]
    close_ = hist['Close'].iloc[-2:]
    high_ = hist['High'].iloc[-2:]
    low_ = hist['Low'].iloc[-2:]
    patterns = []
    # Bullish Engulfing
    if close_.iloc[-2] < open_.iloc[-2] and close_.iloc[-1] > open_.iloc[-1] and close_.iloc[-1] > open_.iloc[-2] and open_.iloc[-1] < close_.iloc[-2]:
        patterns.append('Bullish Engulfing')
    # Bearish Engulfing
    if close_.iloc[-2] > open_.iloc[-2] and close_.iloc[-1] < open_.iloc[-1] and close_.iloc[-1] < open_.iloc[-2] and open_.iloc[-1] > close_.iloc[-2]:
        patterns.append('Bearish Engulfing')
    # Doji
    if abs(close_.iloc[-1] - open_.iloc[-1]) < 0.1 * (high_.iloc[-1] - low_.iloc[-1]):
        patterns.append('Doji')
    # Hammer
    if (high_.iloc[-1] - low_.iloc[-1]) > 3 * abs(open_.iloc[-1] - close_.iloc[-1]) and (close_.iloc[-1] - low_.iloc[-1]) < 0.25 * (high_.iloc[-1] - low_.iloc[-1]):
        patterns.append('Hammer')
    return patterns

def get_default_nse_stocks():
    # Nifty 50 + Nifty Next 50 + some midcaps (can be expanded)
    return [
        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "KOTAKBANK", "ITC",
        "LT", "AXISBANK", "BAJFINANCE", "HCLTECH", "ASIANPAINT", "MARUTI", "SUNPHARMA", "TITAN", "ULTRACEMCO", "WIPRO",
        "POWERGRID", "ONGC", "NTPC", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
        "DIVISLAB", "GRASIM", "HDFCLIFE", "HEROMOTOCO", "HINDALCO", "INDUSINDBK", "M&M", "NESTLEIND", "SBILIFE", "SHREECEM",
        "TECHM", "BAJAJ-AUTO", "BPCL", "BRITANNIA", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "TATACONSUM", "UPL",
        "AMBUJACEM", "APOLLOHOSP", "AUROPHARMA", "BAJAJFINSV", "BANDHANBNK", "BERGEPAINT", "BIOCON", "BOSCHLTD", "CADILAHC",
        "CHOLAFIN", "COLPAL", "DABUR", "DIXON", "DLF", "GAIL", "GLAND", "GODREJCP", "HAVELLS", "ICICIGI", "ICICIPRULI",
        "IGL", "INDIGO", "LUPIN", "MCDOWELL-N", "MOTHERSUMI", "MUTHOOTFIN", "NAUKRI", "PEL", "PIDILITIND", "PIIND",
        "PNB", "RECLTD", "SAIL", "SRF", "SRTRANSFIN", "TORNTPHARM", "TRENT", "TVSMOTOR", "UBL", "VEDL", "VOLTAS", "ZEEL"
    ]

def load_stock_list(category):
    if category == 'Nifty 500':
        try:
            with open('nifty500_stocks.txt', 'r') as f:
                lines = f.readlines()
                # Find the line with actual symbols (skip comments)
                for line in lines:
                    if not line.strip().startswith('#') and line.strip():
                        # Extract symbols from lines like: 'Company Name (SYMBOL), ...'
                        symbols = []
                        for part in line.split(','):
                            if '(' in part and ')' in part:
                                symbol = part.split('(')[-1].split(')')[0].strip()
                                symbols.append(symbol)
                        return symbols
            return []
        except Exception:
            return []
    file_map = {
        'Small': 'small_company_stocks.txt',
        'Mid': 'mid_company_stocks.txt',
        'Large': 'large_company_stocks.txt'
    }
    try:
        with open(file_map[category], 'r') as f:
            stocks = [s.strip().upper() for s in f.read().split(',') if s.strip()]
        return stocks
    except Exception:
        return []

def refresh_stock_lists():
    # You can update this logic to fetch latest lists from an API or database
    small = "RECLTD, SAIL, SRF, SRTRANSFIN, TRENT, TVSMOTOR, UBL, VEDL, VOLTAS, ZEEL, BANDHANBNK, BERGEPAINT, BIOCON, BOSCHLTD, CADILAHC, CHOLAFIN, COLPAL, DABUR, DIXON, GLAND, GODREJCP, HAVELLS, IGL, LUPIN, MCDOWELL-N, MOTHERSUMI, MUTHOOTFIN, NAUKRI, PEL, PIDILITIND, PIIND, PNB, AMBUJACEM, AUROPHARMA, GAIL, SAIL, SRF, SRTRANSFIN, TORNTPHARM, TRENT, TVSMOTOR, UBL, VEDL, VOLTAS, ZEEL"
    mid = "APOLLOHOSP, BAJAJFINSV, DLF, GAIL, GLAND, GODREJCP, HAVELLS, ICICIGI, ICICIPRULI, INDIGO, MCDOWELL-N, MOTHERSUMI, MUTHOOTFIN, NAUKRI, PIDILITIND, PIIND, RECLTD, SRF, SRTRANSFIN, TORNTPHARM, TRENT, TVSMOTOR, UBL, VEDL, VOLTAS, ZEEL"
    large = "RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK, HINDUNILVR, SBIN, BHARTIARTL, KOTAKBANK, ITC, LT, AXISBANK, BAJFINANCE, HCLTECH, ASIANPAINT, MARUTI, SUNPHARMA, TITAN, ULTRACEMCO, WIPRO, POWERGRID, ONGC, NTPC, TATAMOTORS, TATASTEEL, JSWSTEEL, ADANIENT, ADANIGREEN, ADANIPORTS, ADANIPOWER, DIVISLAB, GRASIM, HDFCLIFE, HEROMOTOCO, HINDALCO, INDUSINDBK, M&M, NESTLEIND, SBILIFE, SHREECEM, TECHM, BAJAJ-AUTO, BPCL, BRITANNIA, CIPLA, COALINDIA, DRREDDY, EICHERMOT, TATACONSUM, UPL"
    with open('small_company_stocks.txt', 'w') as f:
        f.write(small)
    with open('mid_company_stocks.txt', 'w') as f:
        f.write(mid)
    with open('large_company_stocks.txt', 'w') as f:
        f.write(large)

# Utility function to calculate RSI
@st.cache_data(show_spinner=False)
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Utility function to calculate Resistance levels
@st.cache_data(show_spinner=False)
def calculate_resistances(hist, window=20):
    highs = hist['High'].rolling(window=window).max()
    resistances = highs[-2:]
    if len(resistances) < 2:
        return None, None
    return resistances.iloc[-1], resistances.iloc[-2]

# Utility function for more technical indicators
# (already cached above)

@st.cache_data(show_spinner=False)
def detect_breakout(hist):
    try:
        hist['EMA20'] = hist['Close'].ewm(span=20).mean()
        hist['EMA50'] = hist['Close'].ewm(span=50).mean()
        hist['EMA200'] = hist['Close'].ewm(span=200).mean()
        last = hist.iloc[-1]
        prev = hist.iloc[-2]
        # EMA crossover
        if last['EMA20'] > last['EMA50'] > last['EMA200'] and prev['EMA20'] <= prev['EMA50']:
            return 'Breakout (EMA Crossover)'
        # Price breaking resistance with volume
        recent_high = hist['High'][-20:].max()
        if last['Close'] > recent_high * 0.995 and last['Volume'] > hist['Volume'][-20:].mean() * 1.2:
            return 'Breakout (Resistance & Volume)'
        return 'No breakout'
    except Exception:
        return 'No breakout'

@st.cache_data(show_spinner=False)
def calculate_targets(hist):
    try:
        recent_lows = hist['Low'][-20:]
        recent_highs = hist['High'][-20:]
        swing_low = recent_lows.min()
        swing_high = recent_highs.max()
        # Fibonacci levels
        diff = swing_high - swing_low
        fib_0_382 = swing_high - 0.382 * diff
        fib_0_618 = swing_high - 0.618 * diff
        return {
            'short_term': swing_high,
            'long_term': swing_high + diff * 0.618,
            'fib_0.382': fib_0_382,
            'fib_0.618': fib_0_618
        }
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def calculate_intrinsic_value(info, growth_rate=0.08, discount_rate=0.12, years=5):
    try:
        fcf = info.get('freeCashflow', None)
        if not fcf:
            fcf = info.get('operatingCashflow', 0) * 0.7  # fallback
        if not fcf or fcf <= 0:
            return None
        intrinsic = 0
        for i in range(1, years+1):
            intrinsic += (fcf * ((1 + growth_rate) ** i)) / ((1 + discount_rate) ** i)
        terminal = (fcf * ((1 + growth_rate) ** years)) * (1 + growth_rate) / (discount_rate - growth_rate)
        terminal /= ((1 + discount_rate) ** years)
        intrinsic += terminal
        shares = info.get('sharesOutstanding', 1)
        return intrinsic / shares
    except Exception:
        return None

def get_action_and_reason(current_price, intrinsic, support_area, resistance_area, fund_score, tech_score):
    action = 'Hold'
    reason = ''
    try:
        if (intrinsic and current_price <= intrinsic * 1.03) or (support_area != '-' and current_price <= float(support_area.replace('₹','')) * 1.03):
            action = 'Buy'
            reason = 'At/near support or intrinsic value'
        elif fund_score >= 6 and tech_score >= 4:
            action = 'Buy'
            reason = 'Strong fundamentals & technicals'
        elif (resistance_area != '-' and current_price >= float(resistance_area.replace('₹','')) * 0.98) or fund_score <= 3 or tech_score <= 2:
            action = 'Sell'
            reason = 'At/near resistance or weak scores'
        else:
            action = 'Hold'
            reason = 'Neutral scores'
    except Exception:
        action = 'Hold'
        reason = 'Neutral scores'
    return action, reason

# ----------------------
# Streamlit App
# ----------------------
st.set_page_config(page_title="Balram Stock Analyzer", layout="wide")

st.title("Balram Stock Analyzer")

# Add session state for selected stock from batch table
if 'show_full_analysis_for' not in st.session_state:
    st.session_state['show_full_analysis_for'] = None

# --- List Search (Batch Analyzer) Section ---
st.markdown('''<div style="background: #e3f2fd; border-left: 6px solid #1976d2; border-radius:12px; padding:1em 1.5em; margin-bottom:1.5em;">
<b>List Search</b> &nbsp; <span style="font-size:0.95em; color:#1976d2;">(Analyze multiple stocks at once: Enter comma-separated NSE symbols)</span></div>''', unsafe_allow_html=True)

# Remove 'Clear Watchlist' button and sector filter selectbox
# --- Sector/Industry Filter and Watchlist Feature ---
sector_options = [
    'All', 'IT', 'Banking', 'Pharma', 'FMCG', 'Auto', 'Energy', 'Metals', 'Infra', 'Consumer', 'Others'
]
# selected_sector = st.selectbox('Filter by Sector/Industry (for List Search)', sector_options, index=0)
selected_sector = 'All'  # Always default to 'All', hide the selectbox

sector_map = {
    'IT': ['TCS', 'INFY', 'HCLTECH', 'WIPRO', 'TECHM'],
    'Banking': ['HDFCBANK', 'ICICIBANK', 'SBIN', 'KOTAKBANK', 'AXISBANK'],
    'Pharma': ['SUNPHARMA', 'CIPLA', 'DRREDDY', 'AUROPHARMA', 'DIVISLAB'],
    'FMCG': ['HINDUNILVR', 'ITC', 'BRITANNIA', 'DABUR', 'COLPAL'],
    'Auto': ['MARUTI', 'M&M', 'TATAMOTORS', 'EICHERMOT', 'BAJAJ-AUTO'],
    'Energy': ['RELIANCE', 'ONGC', 'NTPC', 'POWERGRID', 'BPCL'],
    'Metals': ['TATASTEEL', 'JSWSTEEL', 'HINDALCO', 'VEDL', 'COALINDIA'],
    'Infra': ['LT', 'DLF', 'GRASIM', 'SHREECEM', 'ULTRACEMCO'],
    'Consumer': ['ASIANPAINT', 'NESTLEIND', 'PIDILITIND', 'BERGEPAINT', 'TRENT'],
    'Others': []
}

# Watchlist feature (session state)
if 'watchlist' not in st.session_state:
    st.session_state['watchlist'] = []

# Remove Watchlist header and list from sidebar
# st.sidebar.header('📋 My Watchlist')
# for stock in st.session_state['watchlist']:
#     st.sidebar.write(f'- {stock}')

if selected_sector == 'All':
    multi_input = st.text_input("Stock Symbols (comma-separated, e.g. RELIANCE,TCS,INFY)", "")
    symbols = [s.strip().upper() for s in multi_input.split(",") if s.strip()]
else:
    sector_stocks = sector_map.get(selected_sector, [])
    if selected_sector != 'All' and sector_stocks:
        st.info(f"Stocks in {selected_sector} sector: {', '.join(sector_stocks)}")
        selected_stocks = st.multiselect(f"Select stocks in {selected_sector} sector", sector_stocks, default=sector_stocks)
        symbols = selected_stocks

batch_analysis_ran = False
if st.button("List Search") and symbols:
    with st.spinner("Analyzing batch stocks..."):
        results = []
        for symbol in symbols:
            hist, info = fetch_stock_data(symbol)
            if hist is None or info is None:
                results.append({
                    'Symbol': symbol,
                    'Name': '-',
                    'Fundamental Score': '-',
                    'Technical Score': '-',
                    'Buy Price': '-',
                    'Sell Price': '-',
                    'Support': '-',
                    'Resistance': '-',
                    'Current Price': '-',
                    'Action': 'Data Error',
                    'Reason': 'No data found'
                })
                continue
            # --- Calculate scores (reuse logic) ---
            roe = info.get('returnOnEquity', None)
            debt_equity = info.get('debtToEquity', None)
            pb = info.get('priceToBook', None)
            dividend_yield = info.get('dividendYield', None)
            sales_growth = info.get('revenueGrowth', None)
            profit_growth = info.get('earningsQuarterlyGrowth', None)
            fcf = info.get('freeCashflow', None)
            promoter_holding = info.get('heldPercentInsiders', None)
            rsi = calculate_rsi(hist['Close']).iloc[-1]
            ema20 = hist['Close'].ewm(span=20).mean().iloc[-1]
            ema50 = hist['Close'].ewm(span=50).mean().iloc[-1]
            ema100 = hist['Close'].ewm(span=100).mean().iloc[-1]
            ema200 = hist['Close'].ewm(span=200).mean().iloc[-1]
            techs = calculate_technical_indicators(hist)
            breakout = detect_breakout(hist)
            # Fundamental Score
            fund_score = 0
            if roe and roe > 0.20:
                fund_score += 1
            if debt_equity is not None and debt_equity < 0.3:
                fund_score += 1
            if pb and pb < 3:
                fund_score += 1
            if dividend_yield and dividend_yield > 0.015:
                fund_score += 1
            if sales_growth and sales_growth > 0.12:
                fund_score += 1
            if profit_growth and profit_growth > 0.12:
                fund_score += 1
            if fcf and fcf > 0:
                fund_score += 1
            if promoter_holding and promoter_holding > 0.5:
                fund_score += 1
            # Technical Score
            tech_score = 0
            if rsi and 40 < rsi < 65:
                tech_score += 1
            if ema20 > ema50 > ema100 > ema200:
                tech_score += 2
            if techs['MACD'] > techs['MACD_signal']:
                tech_score += 1
            if techs['ADX'] > 20:
                tech_score += 1
            if breakout.startswith('Breakout'):
                tech_score += 1
            # Price targets and support/resistance
            targets = calculate_targets(hist)
            support_area = '-'
            resistance_area = '-'
            if hist is not None and 'Low' in hist.columns:
                support_area = f"₹{hist['Low'][-20:].min():.2f}"
            if hist is not None and 'High' in hist.columns:
                resistance_area = f"₹{hist['High'][-20:].max():.2f}"
            current_price = hist['Close'][-1]
            intrinsic = calculate_intrinsic_value(info)
            # Always calculate both buy and sell prices
            buy_candidates = []
            if intrinsic and intrinsic < current_price:
                buy_candidates.append(intrinsic * 0.98)
            if support_area != '-' and float(support_area.replace('₹','')) < current_price:
                buy_candidates.append(float(support_area.replace('₹','')))
            buy_candidates.append(current_price)
            best_buy = min(buy_candidates) if buy_candidates else current_price
            sell_candidates = []
            if resistance_area != '-' and float(resistance_area.replace('₹','')) > current_price:
                sell_candidates.append(float(resistance_area.replace('₹','')))
            sell_candidates.append(current_price)
            best_sell = max(sell_candidates) if sell_candidates else current_price
            buy_price = f"₹{best_buy:.2f}"
            sell_price = f"₹{best_sell:.2f}"
            # Consistent Action logic
            action, reason = get_action_and_reason(current_price, intrinsic, support_area, resistance_area, fund_score, tech_score)
            results.append({
                'Symbol': symbol,
                'Name': info.get('shortName', symbol),
                'Fundamental Score': fund_score,
                'Technical Score': tech_score,
                'Buy Price': buy_price,
                'Sell Price': sell_price,
                'Support': support_area,
                'Resistance': resistance_area,
                'Current Price': f"₹{current_price:.2f}",
                'Action': action,
                'Reason': reason
            })
        df = pd.DataFrame(results)
        # --- UI Enhancements ---
        # Color map for scores
        def color_score(val, maxval):
            if val == '-':
                return 'background-color: #eee; color: #888;'
            ratio = val / maxval
            if ratio >= 0.75:
                return 'background-color: #c8e6c9; color: #1b5e20;'
            elif ratio >= 0.5:
                return 'background-color: #fff9c4; color: #f57c00;'
            else:
                return 'background-color: #ffcdd2; color: #b71c1c;'
        def color_action(val):
            if val == 'Buy':
                return 'background-color: #c8e6c9; color: #1b5e20; font-weight:600;'
            elif val == 'Sell':
                return 'background-color: #ffcdd2; color: #b71c1c; font-weight:600;'
            elif val == 'Hold':
                return 'background-color: #fff9c4; color: #f57c00; font-weight:600;'
            else:
                return 'background-color: #eee; color: #888;'
        # Add icons for action
        def action_icon(val):
            if val == 'Buy':
                return '🟢'
            elif val == 'Sell':
                return '🔴'
            elif val == 'Hold':
                return '🟡'
            else:
                return '⚪️'
        df['Action Icon'] = df['Action'].apply(action_icon)
        # Reorder columns for better UX
        col_order = ['Action Icon','Symbol','Name','Fundamental Score','Technical Score','Buy Price','Sell Price','Support','Resistance','Current Price','Action','Reason']
        df = df[col_order]
        #
        buy_count = (df['Action']=='Buy').sum()
        hold_count = (df['Action']=='Hold').sum()
        sell_count = (df['Action']=='Sell').sum()
        st.markdown(f"<b>Summary:</b> <span style='color:#1b5e20;'>🟢 Buy: {buy_count}</span> &nbsp; <span style='color:#f57c00;'>🟡 Hold: {hold_count}</span> &nbsp; <span style='color:#b71c1c;'>🔴 Sell: {sell_count}</span>", unsafe_allow_html=True)
        # --- Custom table with Analyze button in last column ---
        st.markdown('<b>Batch Analysis Results</b>', unsafe_allow_html=True)
        table_cols = list(df.columns) + ['Analyze']
        header_cols = st.columns(len(table_cols))
        for i, col in enumerate(table_cols):
            header_cols[i].markdown(f"<b>{col}</b>", unsafe_allow_html=True)
        # Only show as many rows as there are stocks in the input list
        max_rows = len(symbols)
        for i, (df_idx, row) in enumerate(df.iterrows()):
            if i >= max_rows:
                break
            row_cols = st.columns(len(table_cols))
            for j, col in enumerate(df.columns):
                row_cols[j].write(row[col])
            stable_key = f"analyze_{row['Symbol']}_{df_idx}_{i}_{hash(str(row))}"
            if row_cols[-1].button("Analyze", key=stable_key):
                st.session_state['show_full_analysis_for'] = row['Symbol']
        # If only one stock in list, auto-show full analysis
        if len(symbols) == 1:
            st.session_state['show_full_analysis_for'] = symbols[0]
        # Style table
        styled = df.style.applymap(lambda v: color_score(v,8) if isinstance(v,int) and v<=8 else '', subset=['Fundamental Score']) \
            .applymap(lambda v: color_score(v,6) if isinstance(v,int) and v<=6 else '', subset=['Technical Score']) \
            .applymap(color_action, subset=['Action'])
        st.dataframe(styled, use_container_width=True, height=500)
        # Export button
        st.download_button('Download Table as CSV', df.to_csv(index=False), file_name='list_search_results.csv', mime='text/csv')
        # Best picks
        best = df[df['Action']=='Buy']
        if not best.empty:
            st.success(f"Best Picks: {', '.join(best['Symbol'])}")
        else:
            st.info("No strong buy recommendations in this batch.")
            # Add to watchlist button for each stock
            # Use a unique key for each button by including the row index
            if st.button(f"Add {symbol} to Watchlist", key=f"add_{symbol}_{i}"):
                if symbol not in st.session_state['watchlist']:
                    st.session_state['watchlist'].append(symbol)

# Store only as many rows as there are stocks in the input
        df = df.iloc[:len(symbols)]
        st.session_state['batch_df'] = df
        st.session_state['batch_symbols'] = symbols
        st.session_state['batch_results_ready'] = True
        # Only set show_full_analysis_for if exactly one stock and not already set
        if len(symbols) == 1:
            st.session_state['show_full_analysis_for'] = symbols[0]
        batch_analysis_ran = True
else:
    # --- Render batch table and Analyze buttons if results exist ---
    if not batch_analysis_ran and st.session_state.get('batch_results_ready') and st.session_state.get('batch_df') is not None:
        df = st.session_state['batch_df']
        symbols = st.session_state['batch_symbols']
        st.markdown('<b>Batch Analysis Results</b>', unsafe_allow_html=True)
        table_cols = list(df.columns) + ['Analyze']
        header_cols = st.columns(len(table_cols))
        for i, col in enumerate(table_cols):
            header_cols[i].markdown(f"<b>{col}</b>", unsafe_allow_html=True)
        # Only show as many rows as there are stocks in the input list
        max_rows = len(symbols)
        for i, (df_idx, row) in enumerate(df.iterrows()):
            if i >= max_rows:
                break
            row_cols = st.columns(len(table_cols))
            for j, col in enumerate(df.columns):
                row_cols[j].write(row[col])
            stable_key = f"analyze_{row['Symbol']}_{df_idx}_{i}_{hash(str(row))}"
            if row_cols[-1].button("Analyze", key=stable_key):
                st.session_state['show_full_analysis_for'] = row['Symbol']
        # If only one stock in list, auto-show full analysis
        if len(symbols) == 1 and not st.session_state.get('show_full_analysis_for'):
            st.session_state['show_full_analysis_for'] = symbols[0]

# --- Show full analysis for selected stock (from batch or single) ---
selected_stock = st.session_state.get('show_full_analysis_for', None)
if selected_stock:
    hist, info = fetch_stock_data(selected_stock)
    if hist is None or info is None:
        st.error("Could not fetch data for analysis.")
    else:
        # --- Concise Stock Analysis Summary ---
        roe = info.get('returnOnEquity', None)
        debt_equity = info.get('debtToEquity', None)
        pb = info.get('priceToBook', None)
        dividend_yield = info.get('dividendYield', None)
        sales_growth = info.get('revenueGrowth', None)
        profit_growth = info.get('earningsQuarterlyGrowth', None)
        fcf = info.get('freeCashflow', None)
        promoter_holding = info.get('heldPercentInsiders', None)
        rsi = calculate_rsi(hist['Close']).iloc[-1]
        ema20 = hist['Close'].ewm(span=20).mean().iloc[-1]
        ema50 = hist['Close'].ewm(span=50).mean().iloc[-1]
        ema100 = hist['Close'].ewm(span=100).mean().iloc[-1]
        ema200 = hist['Close'].ewm(span=200).mean().iloc[-1]
        techs = calculate_technical_indicators(hist)
        breakout = detect_breakout(hist)
        # Fundamental Score
        fund_score = 0
        if roe and roe > 0.20:
            fund_score += 1
        if debt_equity is not None and debt_equity < 0.3:
            fund_score += 1
        if pb and pb < 3:
            fund_score += 1
        if dividend_yield and dividend_yield > 0.015:
            fund_score += 1
        if sales_growth and sales_growth > 0.12:
            fund_score += 1
        if profit_growth and profit_growth > 0.12:
            fund_score += 1
        if fcf and fcf > 0:
            fund_score += 1
        if promoter_holding and promoter_holding > 0.5:
            fund_score += 1
        # Technical Score
        tech_score = 0
        if rsi and 40 < rsi < 65:
            tech_score += 1
        if ema20 > ema50 > ema100 > ema200:
            tech_score += 2
        if techs['MACD'] > techs['MACD_signal']:
            tech_score += 1
        if techs['ADX'] > 20:
            tech_score += 1
        if breakout.startswith('Breakout'):
            tech_score += 1
        # Price targets and support/resistance
        sr_20d_support = f"₹{hist['Low'][-20:].min():.2f}" if len(hist) >= 20 else '-'
        sr_20d_resistance = f"₹{hist['High'][-20:].max():.2f}" if len(hist) >= 20 else '-'
        sr_6m_support = f"₹{hist['Low'][-126:].min():.2f}" if len(hist) >= 126 else '-'
        sr_6m_resistance = f"₹{hist['High'][-126:].max():.2f}" if len(hist) >= 126 else '-'
        current_price = hist['Close'][-1]
        intrinsic = calculate_intrinsic_value(info)
        action, reason = get_action_and_reason(current_price, intrinsic, sr_20d_support, sr_20d_resistance, fund_score, tech_score)
        # Calculate val_score before summary section
        val_score = '-' if not intrinsic else f"{(current_price/intrinsic):.2f}x"
        # --- Main Analysis Card ---
        st.markdown(f"""
        <div style='background:#e3f2fd;border-radius:16px;padding:1.5em 2em;margin-bottom:1.5em;border-left:8px solid #1976d2;box-shadow:0 2px 12px #1976d233;'>
        <h3 style='margin-bottom:0.2em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.3em;'>📊</span> Analysis for {selected_stock}</h3>
        <b>Current Price:</b> ₹{current_price:.2f}<br/>
        <b>Intrinsic Value (FCF):</b> {f'₹{intrinsic:.2f}' if intrinsic else 'Not available'}<br/>
        <b>Support (20d):</b> {sr_20d_support} &nbsp; <b>Resistance (20d):</b> {sr_20d_resistance}<br/>
        <b>Support (6m):</b> {sr_6m_support} &nbsp; <b>Resistance (6m):</b> {sr_6m_resistance}<br/>
        <b>Action:</b> <span style='color:#1976d2;font-weight:600;'>{action}</span> - {reason}
        </div>
        """, unsafe_allow_html=True)
        # --- Summary Section ---
        st.markdown(f"""
        <div style='background:linear-gradient(90deg,#fffde7 60%,#fff9c4 100%);border-radius:14px;padding:1.2em 1.5em;margin-bottom:1.2em;border-left:7px solid #fbc02d;box-shadow:0 1px 8px #fbc02d22;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>📝</span> Summary</h4>
        <ul style='font-size:1.13em;line-height:1.7;'>
            <li><b>Recommendation:</b> <span style='color:#1976d2;font-weight:600;'>{action}</span> - {reason}</li>
            <li><b>Current Price:</b> ₹{current_price:.2f}</li>
            <li><b>Intrinsic Value (FCF):</b> {f'₹{intrinsic:.2f}' if intrinsic else 'Not available'}</li>
            <li><b>Valuation Ratio (Price/Intrinsic):</b> {val_score}</li>
            <li><b>Support (20d):</b> {sr_20d_support} | <b>Resistance (20d):</b> {sr_20d_resistance}</li>
            <li><b>Support (6m):</b> {sr_6m_support} | <b>Resistance (6m):</b> {sr_6m_resistance}</li>
            <li><b>Fundamental Score:</b> {fund_score}/8</li>
            <li><b>Technical Score:</b> {tech_score}/6</li>
            <li><b>Breakout Signal:</b> {breakout}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        # --- Momentum & Volatility Section ---
        st.markdown("""
        <div style='background:#e1f5fe;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #0288d1;box-shadow:0 1px 6px #0288d122;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>⚡️</span> Momentum & Volatility</h4>
        """, unsafe_allow_html=True)
        rsi_val = f"{rsi:.2f}" if rsi is not None else 'N/A'
        # Fix: Pass all required arguments (high, low, close) to AverageTrueRange
        atr = ta.volatility.AverageTrueRange(hist['High'], hist['Low'], hist['Close']).average_true_range().iloc[-1]
        bb_width = ta.volatility.BollingerBands(hist['Close']).bollinger_wband().iloc[-1]
        vol_now = hist['Volume'].iloc[-1]
        vol_avg = hist['Volume'].rolling(window=20).mean().iloc[-1]
        st.markdown(f"""
        <ul style='font-size:1.08em;'>
            <li><b>RSI:</b> {rsi_val}</li>
            <li><b>ATR (Volatility):</b> {atr:.2f}</li>
            <li><b>Bollinger Band Width:</b> {bb_width:.2f}</li>
            <li><b>Current Volume:</b> {vol_now:,} | <b>20d Avg:</b> {vol_avg:,.0f}</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        # --- Pattern Detection Section ---
        patterns = detect_candlestick_patterns(hist)
        st.markdown("""
        <div style='background:#f3e5f5;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #8e24aa;box-shadow:0 1px 6px #8e24aa22;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>🕯️</span> Pattern Detection</h4>
        """, unsafe_allow_html=True)
        if patterns:
            st.markdown(f"<b>Detected Patterns:</b> {', '.join(patterns)}", unsafe_allow_html=True)
        else:
            st.markdown("<i>No major candlestick patterns detected in last 2 days.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # --- Risk Factors Section ---
        st.markdown("""
        <div style='background:#ffebee;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #c62828;box-shadow:0 1px 6px #c6282822;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>⚠️</span> Risk Factors</h4>
        """, unsafe_allow_html=True)
        risk_points = []
        if atr > 0.05 * current_price:
            risk_points.append("High volatility (ATR > 5% of price)")
        if vol_now < 0.5 * vol_avg:
            risk_points.append("Low liquidity (Current volume < 50% of avg)")
        if fcf is not None and fcf < 0:
            risk_points.append("Negative Free Cash Flow")
        if profit_growth is not None and profit_growth < 0:
            risk_points.append("Negative profit growth")
        if not risk_points:
            risk_points.append("No major risk factors detected.")
        for pt in risk_points:
            st.write(f"- {pt}")
        st.markdown("</div>", unsafe_allow_html=True)
        # --- Short Term & Long Term Analysis ---
        st.markdown("### Short Term Analysis")
        short_term_points = []
        if breakout.startswith('Breakout'):
            short_term_points.append(f"Breakout detected: {breakout}")
        if rsi is not None and rsi < 40:
            short_term_points.append("RSI is oversold (<40): Possible bounce")
        if rsi is not None and rsi > 65:
            short_term_points.append("RSI is overbought (>65): Possible pullback")
        if ema20 > ema50 > ema100 > ema200:
            short_term_points.append("Strong bullish EMA alignment (20>50>100>200)")
        if techs['MACD'] > techs['MACD_signal']:
            short_term_points.append("MACD is bullish")
        if techs['ADX'] > 20:
            short_term_points.append("ADX indicates strong trend")
        if not short_term_points:
            short_term_points.append("No strong short-term signals. Wait for confirmation.")
        for pt in short_term_points:
            st.write(f"- {pt}")
        st.markdown("### Long Term Analysis")
        long_term_points = []
        if fund_score >= 6:
            long_term_points.append("Strong fundamentals for long-term holding")
        if intrinsic and current_price < intrinsic * 1.05:
            long_term_points.append("Price is near or below intrinsic value")
        if sr_6m_support != '-' and current_price <= float(sr_6m_support.replace('₹','')) * 1.05:
            long_term_points.append("Price is near 6-month support")
        if sr_6m_resistance != '-' and current_price >= float(sr_6m_resistance.replace('₹','')) * 0.98:
            long_term_points.append("Price is near 6-month resistance")
        if not long_term_points:
            long_term_points.append("No strong long-term signals. Consider monitoring.")
        for pt in long_term_points:
            st.write(f"- {pt}")
        # --- Trend Analysis Section ---
        st.markdown("""
        <div style='background:#f1f8e9;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #388e3c;box-shadow:0 1px 6px #388e3c22;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>📈</span> Trend Analysis</h4>
        """, unsafe_allow_html=True)
        # Calculate trend slope (last 30 days)
        close_prices = hist['Close'][-30:]
        x = np.arange(len(close_prices))
        if len(close_prices) > 1:
            slope = np.polyfit(x, close_prices, 1)[0]
            trend = 'Uptrend' if slope > 0 else 'Downtrend' if slope < 0 else 'Sideways'
            st.markdown(f"<b>30-day Trend:</b> <span style='color:#388e3c;font-weight:600;'>{trend}</span> (slope: {slope:.2f})", unsafe_allow_html=True)
        else:
            st.markdown("<i>Not enough data for trend analysis.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # --- Fibonacci Retracement Section ---
        st.markdown("""
        <div style='background:#fff3e0;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #f57c00;box-shadow:0 1px 6px #f57c0022;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>🔢</span> Fibonacci Retracement</h4>
        """, unsafe_allow_html=True)
        fib = calculate_targets(hist)
        if fib:
            st.markdown(f"""
            <ul style='font-size:1.08em;'>
                <li><b>Short Term Target:</b> {fib['short_term']:.2f}</li>
                <li><b>Long Term Target:</b> {fib['long_term']:.2f}</li>
                <li><b>Fib 38.2%:</b> {fib['fib_0.382']:.2f}</li>
                <li><b>Fib 61.8%:</b> {fib['fib_0.618']:.2f}</li>
            </ul>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<i>Not enough data for Fibonacci analysis.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        # Seasonality section hidden as per request
        # --- Seasonality Section (HIDDEN) ---
        # st.markdown("""
        # <div style='background:#e8eaf6;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #3949ab;box-shadow:0 1px 6px #3949ab22;'>
        # <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>📅</span> Seasonality</h4>
        # """, unsafe_allow_html=True)
        # if len(hist) >= 250:
        #     hist['Month'] = hist.index.month
        #     monthly_avg = hist.groupby('Month')['Close'].mean()
        #     st.markdown("<b>Average Monthly Close (last year):</b>", unsafe_allow_html=True)
        #     st.bar_chart(monthly_avg)
        # else:
        #     st.markdown("<i>Not enough data for seasonality analysis.</i>", unsafe_allow_html=True)
        # st.markdown("</div>", unsafe_allow_html=True)
        # --- Score Tiles ---
        score_cols = st.columns(3)
        score_cols[0].markdown(f"""
            <div style='background:#43a047;color:white;padding:1em 0.5em;border-radius:8px;text-align:center;'>
            <b>Fundamental</b><br/><span style='font-size:1.5em;'>{fund_score}/8</span>
            </div>""", unsafe_allow_html=True)
        score_cols[1].markdown(f"""
            <div style='background:#1976d2;color:white;padding:1em 0.5em;border-radius:8px;text-align:center;'>
            <b>Technical</b><br/><span style='font-size:1.5em;'>{tech_score}/6</span>
            </div>""", unsafe_allow_html=True)
        val_score = '-' if not intrinsic else f"{(current_price/intrinsic):.2f}x"
        score_cols[2].markdown(f"""
            <div style='background:#fbc02d;color:white;padding:1em 0.5em;border-radius:8px;text-align:center;'>
            <b>Valuation</b><br/><span style='font-size:1.5em;'>{val_score}</span>
            </div>""", unsafe_allow_html=True)
        # --- EMA + Candlestick Chart ---
        st.markdown("### Price & EMA Trend (Candlestick)")
        import plotly.graph_objs as go
        ema_fig = go.Figure()
        ema_fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Candlestick',
            increasing_line_color='#43a047', decreasing_line_color='#e53935',
            showlegend=True
        ))
        ema_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=20).mean(), mode='lines', name='EMA 20', line=dict(color='blue', width=2)))
        ema_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=50).mean(), mode='lines', name='EMA 50', line=dict(color='red', width=2)))
        ema_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=100).mean(), mode='lines', name='EMA 100', line=dict(color='orange', width=2)))
        ema_fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].ewm(span=200).mean(), mode='lines', name='EMA 200', line=dict(color='green', width=2)))
        ema_fig.update_layout(height=350, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
        st.plotly_chart(ema_fig, use_container_width=True)
        # --- Advanced Fundamentals Tiles ---
        st.markdown("### Advanced Fundamentals")
        adv_metrics = [
            ("ROE > 20%", roe, roe is not None and roe > 0.20),
            ("Debt/Equity < 0.3", debt_equity, debt_equity is not None and debt_equity < 0.3),
            ("P/B < 3", pb, pb is not None and pb < 3),
            ("Dividend Yield > 1.5%", dividend_yield, dividend_yield is not None and dividend_yield > 0.015),
            ("Sales Growth > 12%", sales_growth, sales_growth is not None and sales_growth > 0.12),
            ("Profit Growth > 12%", profit_growth, profit_growth is not None and profit_growth > 0.12),
            ("Free Cash Flow > 0", fcf, fcf is not None and fcf > 0),
            ("Promoter Holding > 50%", promoter_holding, promoter_holding is not None and promoter_holding > 0.5),
        ]
        adv_cols = st.columns(4)
        for i, (label, value, passed) in enumerate(adv_metrics):
            if value is None:
                color = '#bdbdbd'
                text = 'N/A'
            elif passed:
                color = '#43a047'
                text = 'Yes'
            else:
                color = '#e53935'
                text = 'No'
            adv_cols[i%4].markdown(f"""
                <div style='background:{color};color:white;padding:0.7em 1em;border-radius:8px;margin-bottom:0.5em;text-align:center;'>
                    <b>{label}</b><br/>{text} <span style='font-size:0.95em;'>{'' if value is None else f'({value:.2f})'}</span>
                </div>""", unsafe_allow_html=True)
        # --- Technical Breakdown Tiles ---
        st.markdown("### Technical Breakdown")
        tech_metrics = [
            ("RSI 40-65", rsi, rsi is not None and 40 < rsi < 65),
            ("EMA20>EMA50>EMA100>EMA200", f"{ema20:.2f}>{ema50:.2f}>{ema100:.2f}>{ema200:.2f}", ema20 > ema50 > ema100 > ema200),
            ("MACD > Signal", f"{techs['MACD']:.2f} > {techs['MACD_signal']:.2f}" if techs else '-', techs['MACD'] > techs['MACD_signal'] if techs else False),
            ("ADX > 20", techs['ADX'] if techs else '-', techs['ADX'] > 20 if techs else False),
            ("Breakout", breakout, breakout.startswith('Breakout') if breakout else False),
        ]
        tech_cols = st.columns(3)
        for i, (label, value, passed) in enumerate(tech_metrics):
            if value is None or value == '-':
                color = '#bdbdbd'
                text = 'N/A'
            elif passed:
                color = '#1976d2'
                text = 'Yes'
            else:
                color = '#e53935'
                text = 'No'
            tech_cols[i%3].markdown(f"""
                <div style='background:{color};color:white;padding:0.7em 1em;border-radius:8px;margin-bottom:0.5em;text-align:center;'>
                    <b>{label}</b><br/>{text} <span style='font-size:0.95em;'>{'' if value is None else value}</span>
                </div>""", unsafe_allow_html=True)
        # --- Valuation Section ---
        st.markdown("### Valuation")
        st.write(f"Intrinsic Value (FCF): {f'₹{intrinsic:.2f}' if intrinsic else 'Not available'}")
        st.write(f"Current Price: ₹{current_price:.2f}")
        st.write(f"Support (20d): {sr_20d_support} | Resistance (20d): {sr_20d_resistance}")
        st.write(f"Support (6m): {sr_6m_support} | Resistance (6m): {sr_6m_resistance}")
        st.write(f"Valuation Ratio (Price/Intrinsic): {val_score}")
        # --- SWOT Analysis Section ---
        st.markdown("### SWOT Analysis")
        st.info("""
        **Strengths:**
        - Strong fundamentals if most green above
        - Good technicals if most blue above
        
        **Weaknesses:**
        - Red tiles above indicate risk factors
        
        **Opportunities:**
        - If price is near support/intrinsic, may be a good entry
        
        **Threats:**
        - If price is near resistance or technicals are weak, caution advised
        """)
        # Add to watchlist button
        if st.button(f"Add {selected_stock} to Watchlist", key=f"add_{selected_stock}"):
            if selected_stock not in st.session_state['watchlist']:
                st.session_state['watchlist'].append(selected_stock)
                st.success(f"✅ {selected_stock} added to your watchlist!")
        # --- Volatility History Section (HIDDEN) ---
        # st.markdown("""
        # <div style='background:#fbe9e7;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #d84315;box-shadow:0 1px 6px #d8431522;'>
        # <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>📊</span> Volatility History</h4>
        # """, unsafe_allow_html=True)
        # # Show ATR history (last 60 days)
        # atr_series = ta.volatility.AverageTrueRange(hist['High'], hist['Low'], hist['Close']).average_true_range()
        # if len(atr_series) >= 10:
        #     st.line_chart(atr_series[-60:])
        # else:
        #     st.markdown("<i>Not enough data for ATR history.</i>", unsafe_allow_html=True)
        # st.markdown("</div>", unsafe_allow_html=True)
        # --- MACD & Signal Line Section ---
        st.markdown("""
        <div style='background:#e0f2f1;border-radius:12px;padding:1em 1.2em;margin-bottom:1.2em;border-left:6px solid #00897b;box-shadow:0 1px 6px #00897b22;'>
        <h4 style='margin-bottom:0.5em;display:flex;align-items:center;gap:0.5em;'><span style='font-size:1.1em;'>📉</span> MACD & Signal Line</h4>
        """, unsafe_allow_html=True)
        macd = techs['MACD'] if techs else None
        macd_signal = techs['MACD_signal'] if techs else None
        if macd is not None and macd_signal is not None:
            macd_df = pd.DataFrame({'MACD': MACD(hist['Close']).macd(), 'Signal': MACD(hist['Close']).macd_signal()})
            st.line_chart(macd_df[-60:])
        else:
            st.markdown("<i>Not enough data for MACD chart.</i>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
