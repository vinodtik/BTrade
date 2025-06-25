import streamlit as st
from nsepython import nse_optionchain_scrapper
import pandas as pd
import math
from scipy.stats import norm
import time

st.set_page_config(page_title="Live Option Chain (NSE)", layout="wide")
st.markdown("""
    <style>
    .main {background-color: #f8fafc;}
    .stDataFrame {background-color: #fff; border-radius: 10px;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 6px; font-weight: 600; font-size: 0.95em; padding: 0.25em 1em; margin: 0.1em; min-width: 90px; min-height: 32px;}
    .stButton>button:active {background-color: #1e40af;}
    .stMetric {background-color: #f1f5f9; border-radius: 10px;}
    .section-card {background: linear-gradient(90deg,#60a5fa,#a7f3d0); color:#222; border-radius:12px; padding:1.2em; margin-bottom:1.2em; font-size:1.15em; box-shadow: 0 2px 8px #0001;}
    .top-table {background: #e0e7ff; border-radius: 12px; padding: 1em; margin-bottom: 1.2em; box-shadow: 0 2px 8px #0001;}
    .refresh-controls {margin-bottom: 1em;}
    .stTextInput>div>input {border-radius: 8px; border: 1.5px solid #2563eb; font-size: 1.1em;}
    .stNumberInput>div>input {border-radius: 8px; border: 1.5px solid #2563eb; font-size: 1.1em;}
    </style>
""", unsafe_allow_html=True)

st.title("📈 Live Option Chain")

symbol = st.text_input("Enter NSE Symbol (e.g., NIFTY, BANKNIFTY, RELIANCE):", value="NIFTY")

# --- Auto-refresh controls ---
if 'autorefresh_on' not in st.session_state:
    st.session_state['autorefresh_on'] = False

st.markdown("<div style='display: flex; gap: 0.5em; align-items: center; margin-bottom: 1em;'>", unsafe_allow_html=True)
refresh_interval = st.number_input("⏱️ Auto-refresh interval (seconds, 0 = off)", min_value=0, max_value=300, value=0, step=1, key='refresh_interval')
start, stop, fetch = st.columns([1,1,1])
with start:
    if st.button("▶️ Start", key='start_btn'):
        st.session_state['autorefresh_on'] = True
with stop:
    if st.button("⏹️ Stop", key='stop_btn'):
        st.session_state['autorefresh_on'] = False
with fetch:
    fetch_clicked = st.button("🔄 Fetch", key='fetch_btn')
st.markdown("</div>", unsafe_allow_html=True)

# --- Helper: Calculate Entry, Stop Loss, Target ---
def calculate_trade_levels(ltp, risk_percent=20, reward_ratio=2):
    entry = round(ltp, 2)
    stop_loss = round(entry * (1 - risk_percent / 100), 2)
    target = round(entry + ((entry - stop_loss) * reward_ratio), 2)
    return entry, stop_loss, target

# --- Helper: Calculate Greeks ---
def calculate_greeks(option_type, S, K, T, r, iv):
    if T <= 0 or iv <= 0 or S <= 0 or K <= 0:
        return {g: 0.0 for g in ['Delta','Gamma','Theta','Vega','Rho']}
    d1 = (math.log(S/K) + (r + 0.5*iv**2)*T) / (iv*math.sqrt(T))
    d2 = d1 - iv*math.sqrt(T)
    if option_type == 'CALL':
        delta = norm.cdf(d1)
        theta = (-S*norm.pdf(d1)*iv/(2*math.sqrt(T)) - r*K*math.exp(-r*T)*norm.cdf(d2))/365
        rho = K*T*math.exp(-r*T)*norm.cdf(d2)/100
    else:
        delta = -norm.cdf(-d1)
        theta = (-S*norm.pdf(d1)*iv/(2*math.sqrt(T)) + r*K*math.exp(-r*T)*norm.cdf(-d2))/365
        rho = -K*T*math.exp(-r*T)*norm.cdf(-d2)/100
    gamma = norm.pdf(d1)/(S*iv*math.sqrt(T))
    vega = S*norm.pdf(d1)*math.sqrt(T)/100
    return {
        'Delta': delta,
        'Gamma': gamma,
        'Theta': theta,
        'Vega': vega,
        'Rho': rho
    }

# --- Helper: OI Behavior ---
def analyze_oi_behavior(oi_change, ltp, avg_ltp):
    if oi_change > 0 and ltp > avg_ltp:
        return 'Long Build-up'
    elif oi_change < 0 and ltp > avg_ltp:
        return 'Short Covering'
    elif oi_change > 0 and ltp < avg_ltp:
        return 'Short Build-up'
    elif oi_change < 0 and ltp < avg_ltp:
        return 'Long Unwinding'
    else:
        return 'Neutral'

# --- Helper: Volume Strength ---
def analyze_volume_strength(volume, avg_volume):
    if volume > avg_volume * 1.2:
        return 'Strong'
    elif volume < avg_volume * 0.8:
        return 'Weak'
    else:
        return 'Normal'

# --- Auto-refresh logic using st_autorefresh ---
if refresh_interval > 0 and st.session_state['autorefresh_on']:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=refresh_interval * 1000, key="autorefresh")

if fetch_clicked or (refresh_interval > 0 and st.session_state['autorefresh_on']):
    try:
        data = nse_optionchain_scrapper(symbol.upper())
        ce = pd.DataFrame(data['records']['data'])[['strikePrice', 'CE']]
        pe = pd.DataFrame(data['records']['data'])[['strikePrice', 'PE']]
        ce = ce.dropna().reset_index(drop=True)
        pe = pe.dropna().reset_index(drop=True)
        ce_df = pd.json_normalize(ce['CE'])
        pe_df = pd.json_normalize(pe['PE'])
        ce_df['Type'] = 'CALL'
        pe_df['Type'] = 'PUT'
        option_df = pd.concat([ce_df, pe_df], ignore_index=True)
        option_df = option_df[option_df['lastPrice'] > 0]
        spot = data['records']['underlyingValue']
        st.markdown(f"<div class='section-card'><b>Spot Price:</b> {spot}</div>", unsafe_allow_html=True)
        option_df['Moneyness'] = abs(option_df['strikePrice'] - spot) / spot
        option_df['OI Change'] = option_df['changeinOpenInterest']
        option_df['Volume'] = option_df['totalTradedVolume']
        option_df['Bid'] = option_df['bidprice']
        option_df['Ask'] = option_df['askPrice']
        option_df['IV'] = option_df['impliedVolatility']
        avg_ltp = option_df['lastPrice'].mean()
        avg_volume = option_df['Volume'].mean()
        # --- Add analytics columns ---
        option_df['OI Behavior'] = option_df.apply(lambda row: analyze_oi_behavior(row['OI Change'], row['lastPrice'], avg_ltp), axis=1)
        option_df['Volume Strength'] = option_df['Volume'].apply(lambda v: analyze_volume_strength(v, avg_volume))
        # Greeks
        T = 2/365  # Assume 2 days to expiry for demo; can be improved
        r = 0.06
        option_df[['Delta','Gamma','Theta','Vega','Rho']] = option_df.apply(
            lambda row: pd.Series(calculate_greeks(row['Type'], spot, row['strikePrice'], T, r, row['IV']/100 if row['IV'] > 0 else 0.18)), axis=1)
        # --- Enhanced Top Recommendations Logic ---
        option_df['Volume Score'] = option_df['Volume'] / option_df['Volume'].max()
        option_df['OI Score'] = option_df['OI Change'] / option_df['OI Change'].abs().max()
        option_df['Bid-Ask Spread'] = option_df['Ask'] - option_df['Bid']
        option_df['Spread Score'] = 1 - (option_df['Bid-Ask Spread'] / option_df['Bid-Ask Spread'].max())
        option_df['Composite Score'] = (
            (1 - option_df['Moneyness']) * 0.4 +
            option_df['OI Score'] * 0.25 +
            option_df['Volume Score'] * 0.25 +
            option_df['Spread Score'] * 0.1
        )
        # --- Add Confidence ---
        def calculate_confidence(row):
            score = 0
            if row['OI Behavior'] == 'Long Build-up':
                score += 2
            elif row['OI Behavior'] == 'Short Covering':
                score += 1
            if row['Volume Strength'] == 'Strong':
                score += 2
            elif row['Volume Strength'] == 'Normal':
                score += 1
            if 0.35 < abs(row['Delta']) < 0.65:
                score += 1
            if abs(row['Theta']) < 0.02:
                score += 1
            if abs(row['Vega']) > 0.05:
                score += 1
            if row['Moneyness'] > 0.05:
                score -= 1
            score = max(1, min(score, 5))
            return ['Very Low','Low','Moderate','High','Very High'][score-1]
        option_df['Confidence'] = option_df.apply(calculate_confidence, axis=1)
        # --- Top Recommendations Table ---
        top_recs = option_df.sort_values('Composite Score', ascending=False).head(2).copy()
        top_recs[['Entry', 'Stop Loss', 'Target']] = top_recs.apply(
            lambda row: calculate_trade_levels(row['lastPrice']), axis=1, result_type='expand')
        # Add Action column (BUY/AVOID) in 3rd position
        def get_action(row):
            oi_behavior = row['OI Behavior'] if 'OI Behavior' in row else row['oi_behavior'] if 'oi_behavior' in row else ''
            volume_strength = row['Volume Strength'] if 'Volume Strength' in row else row['volume_strength'] if 'volume_strength' in row else ''
            if oi_behavior in ['Long Build-up', 'Short Covering'] and volume_strength == 'Strong':
                return 'BUY'
            else:
                return 'AVOID'
        top_recs['Action'] = top_recs.apply(get_action, axis=1)
        display_cols = ['Strike Price', 'Option Type', 'Action', 'LTP', 'Bid', 'Ask', 'OI Change', 'Volume', 'Entry', 'Stop Loss', 'Target', 'Delta', 'Gamma', 'Theta', 'Vega', 'Rho', 'OI Behavior', 'Volume Strength', 'Confidence']
        top_recs = top_recs.rename(columns={
            'strikePrice': 'Strike Price',
            'Type': 'Option Type',
            'lastPrice': 'LTP'
        })[display_cols]
        st.markdown("<div class='top-table'>", unsafe_allow_html=True)
        st.subheader("Top Recommendations (Live, Full Analysis)")
        st.dataframe(top_recs)
        st.markdown("</div>", unsafe_allow_html=True)
        # --- Best OTM Call Table ---
        otm_calls = option_df[(option_df['Type'] == 'CALL') & (option_df['Moneyness'] > 0.03)]
        if not otm_calls.empty:
            best_otm_call = otm_calls.sort_values('Composite Score', ascending=False).head(1).copy()
            best_otm_call[['Entry', 'Stop Loss', 'Target']] = best_otm_call.apply(
                lambda row: calculate_trade_levels(row['lastPrice']), axis=1, result_type='expand')
            best_otm_call['Action'] = best_otm_call.apply(get_action, axis=1)
            best_otm_call = best_otm_call.rename(columns={
                'strikePrice': 'Strike Price',
                'Type': 'Option Type',
                'lastPrice': 'LTP'
            })[display_cols]
            st.markdown("<div class='top-table'>", unsafe_allow_html=True)
            st.subheader("Best OTM Call (Based on Analysis)")
            st.dataframe(best_otm_call)
            st.markdown("</div>", unsafe_allow_html=True)
        # --- SUMMARY CARD ---
        best = top_recs.iloc[0]
        # Use 'IV' if present, else fallback to impliedVolatility or 0.18
        iv_val = best['IV'] if 'IV' in best and pd.notnull(best['IV']) else best['LTP'] if 'impliedVolatility' in best and pd.notnull(best['impliedVolatility']) else 0.18
        strike_val = best['Strike Price'] if 'Strike Price' in best else best['strikePrice'] if 'strikePrice' in best else None
        opt_type_val = best['Option Type'] if 'Option Type' in best else best['Type'] if 'Type' in best else ''
        summary_html = """
        <div class='section-card'>
            <b>Summary:</b><br>
        """
        summary_html += f"<b>Which Strike to Buy:</b> {strike_val} ({opt_type_val})<br>"
        summary_html += f"<b>Entry:</b> {best['Entry']} &nbsp; <b>Stop Loss:</b> {best['Stop Loss']} &nbsp; <b>Target:</b> {best['Target']}<br>"
        # Strategy Suggestion
        oi_behavior = best['OI Behavior'] if 'OI Behavior' in best else best['oi_behavior'] if 'oi_behavior' in best else ''
        volume_strength = best['Volume Strength'] if 'Volume Strength' in best else best['volume_strength'] if 'volume_strength' in best else ''
        if oi_behavior == 'Long Build-up' and volume_strength == 'Strong':
            strategy = 'Momentum Buy'
        elif oi_behavior == 'Short Covering':
            strategy = 'Short Covering Play'
        elif oi_behavior == 'Short Build-up':
            strategy = 'Short Build-up (Avoid Buy)'
        else:
            strategy = 'Check Manually'
        summary_html += f"<b>Strategy Suggestion:</b> {strategy}<br>"
        rr = round((best['Target'] - best['Entry']) / (best['Entry'] - best['Stop Loss']), 2) if (best['Entry'] - best['Stop Loss']) != 0 else 'N/A'
        summary_html += f"<b>Risk/Reward:</b> {rr}<br>"
        expected_move = round(spot * (iv_val/100) * math.sqrt(T), 2)
        summary_html += f"<b>Expected Move (till expiry):</b> {expected_move}<br>"
        summary_html += "</div>"
        st.markdown(summary_html, unsafe_allow_html=True)

        # --- Probability of Expiring ITM/OTM for each strike ---
        def prob_itm(row):
            S = spot
            K = row['Strike Price'] if 'Strike Price' in row else row['strikePrice'] if 'strikePrice' in row else None
            iv = row['IV'] if 'IV' in row and pd.notnull(row['IV']) else row['impliedVolatility'] if 'impliedVolatility' in row and pd.notnull(row['impliedVolatility']) else 18.0
            iv = iv/100 if iv > 0 else 0.18
            opt_type = row['Option Type'] if 'Option Type' in row else row['Type'] if 'Type' in row else ''
            if T <= 0 or iv <= 0 or S <= 0 or K is None or K <= 0:
                return 0.0
            d2 = (math.log(S/K) + (0.06 - 0.5*iv**2)*T) / (iv*math.sqrt(T))
            if opt_type == 'CALL':
                return round(1 - norm.cdf(d2), 2)
            else:
                return round(norm.cdf(-d2), 2)
        option_df['Prob ITM'] = option_df.apply(prob_itm, axis=1)
        option_df['Prob OTM'] = 1 - option_df['Prob ITM']

        # --- Full Option Chain Table ---
        # st.subheader("Full Option Chain Data")
        # st.dataframe(option_df)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
