import streamlit as st
import pandas as pd
import re
from loader import load_option_chain_csv
from strategy import filter_liquid_options, filter_proximity_to_spot
from recommender import recommend_trades
from datetime import datetime
from config import MIN_OI, MIN_VOLUME, MAX_BID_ASK_SPREAD, PROXIMITY_TO_SPOT

@st.cache_data(show_spinner=False)
def process_file(uploaded_file, lower_strike=None, upper_strike=None):
    try:
        # Load the data
        df = load_option_chain_csv(uploaded_file)
        if df.empty:
            st.error("No data could be loaded from the CSV file. Please check the file format.")
            return pd.DataFrame()
            
        # Update all record counts to show no filtering
        total_records = len(df)
        st.session_state.debug_info['raw_records'] = total_records
        st.session_state.debug_info['after_liquidity'] = total_records
        st.session_state.debug_info['after_proximity'] = total_records
        
        # Log data stats
        print(f"\nData statistics:")
        print(f"Total rows: {total_records}")
        print(f"Unique strikes: {df['Strike Price'].nunique()}")
        print(f"Strike range: {df['Strike Price'].min()} to {df['Strike Price'].max()}")
        print(f"Spot price: {df['Spot Price'].iloc[0]}")
        
        # Only show processing message for new file uploads (cached results won't show this)
        if uploaded_file.name not in st.session_state.get('processed_files', set()):
            st.info("Processing option chain data...", icon="ℹ️")
            if 'processed_files' not in st.session_state:
                st.session_state.processed_files = set()
            st.session_state.processed_files.add(uploaded_file.name)
        
        # Apply strike range filter if provided
        if lower_strike is not None and upper_strike is not None:
            df_range = df[
                (df['Strike Price'] >= lower_strike) & 
                (df['Strike Price'] <= upper_strike)
            ].copy()
            
            if df_range.empty:
                st.warning("No data in selected strike range. Showing all available strikes.")
                return df
            return df_range
            
        return df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        print(f"Error details:\n{traceback.format_exc()}")
        return pd.DataFrame()

def analyze_market_sentiment(df):
    if df.empty:
        return 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', [], []
    call_oi = df[df['Option Type'] == 'CALL'].groupby('Strike Price')['OI'].sum()
    put_oi = df[df['Option Type'] == 'PUT'].groupby('Strike Price')['OI'].sum()
    max_call_oi_strike = call_oi.idxmax() if not call_oi.empty else None
    max_put_oi_strike = put_oi.idxmax() if not put_oi.empty else None
    spot = df['Spot Price'].iloc[0] if not df.empty else 0
    support = max_put_oi_strike
    resistance = max_call_oi_strike
    range_width = resistance - support if resistance and support else 'N/A'
    if spot and support and spot < support:
        sentiment = 'Bearish'
    elif spot and resistance and spot > resistance:
        sentiment = 'Bullish'
    elif spot:
        sentiment = 'Neutral'
    else:
        sentiment = 'N/A'
    range_desc = f"{support} - {resistance} (width: {range_width})"
    oi_change = df.groupby(['Strike Price'])['Change in OI'].sum()
    build_up_zones = oi_change.sort_values(ascending=False).head(2)
    unwinding_zones = oi_change.sort_values().head(2)
    build_up_list = [(strike, change) for strike, change in build_up_zones.items() if change > 0]
    unwinding_list = [(strike, change) for strike, change in unwinding_zones.items() if change < 0]
    return sentiment, support, resistance, range_width, range_desc, build_up_list, unwinding_list

def get_max_pain(df):
    if df.empty:
        return 'N/A'
    call_oi = df[df['Option Type'] == 'CALL'].groupby('Strike Price')['OI'].sum()
    put_oi = df[df['Option Type'] == 'PUT'].groupby('Strike Price')['OI'].sum()
    total_oi = call_oi + put_oi
    if total_oi.empty:
        return 'N/A'
    return total_oi.idxmax()

def get_expiry_from_filename(filename):
    m = re.search(r'(\d{1,2}-[A-Za-z]{3}-\d{4})', filename)
    return str(datetime.strptime(m.group(1), '%d-%b-%Y').date()) if m else 'N/A'

def add_analytics_columns(df):
    if df.empty:
        df['PCR'] = []
        df['OI Change %'] = []
        df['IV Rank'] = []
        df['Unusual Activity'] = []
        df['Bid-Ask Spread'] = []
        df['ITM/ATM/OTM'] = []
        return df
    
    # Add PCR calculation
    call_oi = df[df['Option Type'] == 'CALL'].set_index('Strike Price')['OI']
    put_oi = df[df['Option Type'] == 'PUT'].set_index('Strike Price')['OI']
    def safe_get(series, key, default=0):
        val = series.get(key, default)
        if isinstance(val, pd.Series):
            if not val.empty:
                return val.iloc[0]
            else:
                return default
        return val
    df['PCR'] = df['Strike Price'].map(
        lambda x: safe_get(put_oi, x, 0) / safe_get(call_oi, x, 1) if safe_get(call_oi, x, 1) != 0 else 0
    )
    
    # Add OI Change %
    df['OI Change %'] = df.apply(lambda row: (row['Change in OI'] / row['OI'] * 100) if row['OI'] else 0, axis=1)
    
    # Add IV Rank with error handling
    if 'IV' in df.columns:
        min_iv = df['IV'].min()
        max_iv = df['IV'].max()
        df['IV Rank'] = df['IV'].apply(lambda x: (x - min_iv) / (max_iv - min_iv) if max_iv > min_iv else 0)
    else:
        df['IV'] = 0
        df['IV Rank'] = 0
    
    # Add Unusual Activity
    median_oi_change = df['OI Change %'].median()
    if 'Volume' in df.columns:
        median_vol = df['Volume'].median()
        df['Unusual Activity'] = df.apply(lambda row: 'Yes' if abs(row['OI Change %']) > 2 * abs(median_oi_change) or row['Volume'] > 2 * median_vol else 'No', axis=1)
    else:
        df['Volume'] = 0
        df['Unusual Activity'] = df.apply(lambda row: 'Yes' if abs(row['OI Change %']) > 2 * abs(median_oi_change) else 'No', axis=1)
    
    # Add Bid-Ask Spread
    if 'Ask' in df.columns and 'Bid' in df.columns:
        df['Bid-Ask Spread'] = df['Ask'] - df['Bid']
    else:
        df['Bid-Ask Spread'] = 0
    
    # Add ITM/ATM/OTM classification
    spot = df['Spot Price'].iloc[0]
    def moneyness(row):
        if row['Option Type'] == 'CALL':
            if row['Strike Price'] < spot:
                return 'ITM'
            elif abs(row['Strike Price'] - spot) < 0.01 * spot:
                return 'ATM'
            else:
                return 'OTM'
        else:
            if row['Strike Price'] > spot:
                return 'ITM'
            elif abs(row['Strike Price'] - spot) < 0.01 * spot:
                return 'ATM'
            else:
                return 'OTM'
    df['ITM/ATM/OTM'] = df.apply(moneyness, axis=1)
    return df

# Set page config
st.set_page_config(page_title="Balram Analysis", layout="wide")

# Add CSS
st.markdown("""

<style>
    .main {background-color: #f8fafc;}
    .stDataFrame {background-color: #fff; border-radius: 8px;}
    .stButton>button {background-color: #2563eb; color: white; border-radius: 6px;}
    .stMetric {background-color: #f1f5f9; border-radius: 8px;}
    .sentiment-box {background: linear-gradient(90deg,#fbbf24,#f87171); color:#222; border-radius:10px; padding:1em; margin-bottom:1em; font-size:1.2em;}
    .range-box {background: linear-gradient(90deg,#60a5fa,#a7f3d0); color:#222; border-radius:10px; padding:1em; margin-bottom:1em; font-size:1.1em;}
    .debug-box {background: linear-gradient(90deg,#f1f5f9,#e2e8f0); color:#222; border-radius:10px; padding:1em; margin:1em 0; font-size:0.9em;}
    .metric-card {background: white; padding: 10px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 5px 0;}
    .view-mode-trading {background: #f0fdf4; border-radius: 8px; padding: 0.5em; margin-bottom: 1em;}
    .view-mode-analysis {background: #f0f9ff; border-radius: 8px; padding: 0.5em; margin-bottom: 1em;}
</style>

""", unsafe_allow_html=True)

# Initialize debug info
if 'debug_info' not in st.session_state:
    st.session_state.debug_info = {
        'raw_records': 0,
        'after_liquidity': 0,
        'after_proximity': 0,
        'final_records': 0
    }

uploaded_file = st.file_uploader("📁 Upload Option Chain CSV", type=["csv"])

if uploaded_file:
    # Clear previous debug info
    st.session_state.debug_info = {
        'raw_records': 0,
        'after_liquidity': 0,
        'after_proximity': 0,
        'final_records': 0
    }
    
    # Process file and get data
    df = process_file(uploaded_file)
    
    if df.empty:
        st.warning("No data after filtering. Please check your filters or upload a different file.")
    else:
        # Add view mode selector
        st.markdown("""
        <div style='background: linear-gradient(90deg,#e0f2fe,#dbeafe); color:#222; border-radius:10px; padding:1em; margin-bottom:1em;'>
            <h3 style='margin:0; font-size:1.1em;'>📊 Option Chain Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats about the data
        spot_price = float(df['Spot Price'].iloc[0])
        total_call_oi = df[df['Option Type'] == 'CALL']['OI'].sum()
        total_put_oi = df[df['Option Type'] == 'PUT']['OI'].sum()
        pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0
        
        # Show key metrics in a clean layout
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("PCR (Put-Call Ratio)", f"{pcr:.2f}", 
                     delta="Bullish" if pcr > 1.5 else "Bearish" if pcr < 0.5 else "Neutral")
        with metrics_col2:
            atm_strike = df.iloc[(df['Strike Price'] - spot_price).abs().idxmin()]['Strike Price']
            st.metric("ATM Strike", f"{atm_strike:,.2f}")
        with metrics_col3:
            iv_percentile = df['IV'].quantile(0.5)
            st.metric("Median IV", f"{iv_percentile:.1f}%")
        
        # Add strike range selector with improved UI
        st.markdown("""
        <div style='background: linear-gradient(90deg,#e0f2fe,#dbeafe); color:#222; border-radius:10px; padding:1em; margin-bottom:1em;'>
            <h3 style='margin:0; font-size:1.1em;'>🎯 Strike Range Selection</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get min and max strikes
        min_strike = float(df['Strike Price'].min())
        max_strike = float(df['Strike Price'].max())
        spot_price = float(df['Spot Price'].iloc[0])
        
        # Default range: 5% above and below spot price
        default_range = (spot_price * 0.05)
        default_min = max(min_strike, spot_price - default_range)
        default_max = min(max_strike, spot_price + default_range)
        
        # Create strike range selector
        col1, col2 = st.columns(2)
        with col1:
            lower_strike = st.number_input(
                "Lower Strike",
                min_value=min_strike,
                max_value=max_strike,
                value=default_min,
                step=50.0,
                format="%.2f"
            )
        with col2:
            upper_strike = st.number_input(
                "Upper Strike",
                min_value=min_strike,
                max_value=max_strike,
                value=default_max,
                step=50.0,
                format="%.2f"
            )
        
        # Add helper text
        st.caption("💡 Tip: Default range is set to ±5% of spot price. Adjust the range to focus on specific strikes.")
        
        # Show selected range info
        strikes_in_range = len(df[(df['Strike Price'] >= lower_strike) & (df['Strike Price'] <= upper_strike)])
        st.markdown(f"""
        <div style='background: linear-gradient(90deg,#f0fdf4,#ecfdf5); color:#222; border-radius:10px; padding:0.5em; margin-bottom:1em; font-size:0.9em;'>
            <b>Selected Range:</b> {lower_strike:.2f} to {upper_strike:.2f} ({strikes_in_range} strikes) |
            <b>Distance from Spot:</b> -{abs(spot_price - lower_strike):.2f} to +{abs(upper_strike - spot_price):.2f}
        </div>
        """, unsafe_allow_html=True)
        
        # Filter dataframe by selected strike range
        df_filtered = df[(df['Strike Price'] >= lower_strike) & (df['Strike Price'] <= upper_strike)].copy()
        
        if df_filtered.empty:
            st.warning("No data in selected strike range. Please adjust the range.")
        else:
            df = df_filtered  # Use filtered dataframe for all subsequent analysis
            df = add_analytics_columns(df)
            sentiment, support, resistance, range_width, range_desc, build_up_list, unwinding_list = analyze_market_sentiment(df)
            expiry = get_expiry_from_filename(uploaded_file.name)
            max_pain = get_max_pain(df)
            st.markdown(f"<div class='sentiment-box'><b>Market Sentiment:</b> {sentiment} &nbsp; <b>Expiry:</b> {expiry}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='range-box'><b>Range:</b> {range_desc} &nbsp; <b>Max Pain:</b> {max_pain}</div>", unsafe_allow_html=True)
            # --- Show merged OI Build-up/Unwinding Zones: CALL left, Strike center, PUT right ---
            def style_oi_merged_table(df):
                def highlight(val):
                    if pd.isna(val):
                        return ''
                    color = '#d1fae5' if val > 0 else '#fee2e2'
                    return f'background-color: {color}; color: #222;'
                return df.style.applymap(highlight, subset=['CALL OI Change', 'PUT OI Change'])

            # Get top 2 build-up and unwinding for CALL and PUT
            build_up_call = df[df['Option Type'] == 'CALL'].groupby('Strike Price')['Change in OI'].sum().sort_values(ascending=False).head(2)
            build_up_put = df[df['Option Type'] == 'PUT'].groupby('Strike Price')['Change in OI'].sum().sort_values(ascending=False).head(2)
            unwinding_call = df[df['Option Type'] == 'CALL'].groupby('Strike Price')['Change in OI'].sum().sort_values().head(2)
            unwinding_put = df[df['Option Type'] == 'PUT'].groupby('Strike Price')['Change in OI'].sum().sort_values().head(2)

            # Merge build-up and unwinding into one table
            strikes = sorted(set(build_up_call.index).union(build_up_put.index).union(unwinding_call.index).union(unwinding_put.index))
            merged = []
            for strike in strikes:
                merged.append({
                    'CALL OI Change': build_up_call.get(strike) if build_up_call.get(strike, 0) > 0 else unwinding_call.get(strike) if unwinding_call.get(strike, 0) < 0 else None,
                    'Strike Price': strike,
                    'PUT OI Change': build_up_put.get(strike) if build_up_put.get(strike, 0) > 0 else unwinding_put.get(strike) if unwinding_put.get(strike, 0) < 0 else None
                })
            merged_df = pd.DataFrame(merged)
            st.markdown("""
            <div style='background: linear-gradient(90deg,#f0f9ff,#e0f2fe); color:#222; border-radius:10px; padding:1em; margin-bottom:1em;'>
                <h3 style='margin:0; font-size:1.1em;'>📊 Open Interest Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background:#fff; padding:0.5em; border-radius:8px; margin-bottom:1em;'>
                <p style='color:#475569; margin:0 0 0.5em 0;'><b>OI Build-up:</b> Green highlighting indicates fresh positions being built</p>
                <p style='color:#475569; margin:0;'><b>OI Unwinding:</b> Red highlighting shows positions being closed</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(style_oi_merged_table(merged_df))
            
            # Add interpretation of OI data
            if not merged_df.empty:
                calls_buildup = merged_df['CALL OI Change'].max() if not merged_df['CALL OI Change'].isna().all() else 0
                puts_buildup = merged_df['PUT OI Change'].max() if not merged_df['PUT OI Change'].isna().all() else 0
                
                interpretation = ""
                if calls_buildup > 0 and puts_buildup > 0:
                    interpretation = "Both CALLS and PUTS showing build-up indicates high volatility expectations"
                elif calls_buildup > puts_buildup:
                    interpretation = "Stronger CALL build-up suggests bearish sentiment (resistance forming)"
                elif puts_buildup > calls_buildup:
                    interpretation = "Stronger PUT build-up suggests bullish sentiment (support forming)"
                
                if interpretation:
                    st.markdown(f"""
                    <div style='background:#f8fafc; padding:0.8em; border-radius:8px; margin-top:0.5em;'>
                        <p style='color:#1e293b; margin:0;'><b>💡 Interpretation:</b> {interpretation}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("---")
            recommendations = recommend_trades(df)
            df_rec = pd.DataFrame(recommendations)
            
            # Ensure required columns exist
            required_cols = ['Action', 'Confidence', 'Volume Strength']
            for col in required_cols:
                if col not in df_rec.columns:
                    df_rec[col] = 'Unknown'  # Add missing columns with default value
            
            # Section: Top Recommendations Table (Buy actions with Very High confidence only)
            top_recs = df_rec[
                (df_rec['Action'] == 'BUY') & 
                (df_rec['Confidence'] == 'Very High')
            ].sort_values('Volume Strength', ascending=False)
            
            st.markdown("""
            <div style='background: linear-gradient(90deg,#3b82f6,#2563eb); color:white; border-radius:10px; padding:0.8em; margin:1em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                <h3 style='margin:0; font-size:1.2em; font-weight:600;'>🎯 Top Trading Opportunities</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if not top_recs.empty:
                # Create grid container
                st.markdown("""
                <style>
                .trade-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1rem;
                    margin: 1rem 0;
                }
                </style>
                <div class="trade-grid">
                """, unsafe_allow_html=True)
                
                for _, rec in top_recs.iterrows():
                    rr_ratio = ((rec['Target Price'] - rec['Entry Price']) / (rec['Entry Price'] - rec['Stop Loss']))
                    st.markdown(f"""
                        <div style='background:white; border-radius:8px; border-left:4px solid {"#059669" if rec["Confidence"] == "Very High" else "#0284c7"}; 
                             box-shadow:0 1px 3px rgba(0,0,0,0.1); padding:0.8em; height:100%; display:flex; flex-direction:column;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.5em;'>
                            <span style='font-size:1.1em; font-weight:600; color:#1e293b;'>{rec['Option Type']} {rec['Strike Price']}</span>
                            <span style='background:{"#dcfce7" if rec["Confidence"] == "Very High" else "#dbeafe"}; 
                                  color:{"#059669" if rec["Confidence"] == "Very High" else "#2563eb"}; 
                                  padding:0.1em 0.4em; border-radius:4px; font-size:0.8em;'>{rec['Confidence']}</span>
                        </div>
                        <div style='display:grid; grid-template-columns:repeat(3, 1fr); gap:0.5em; margin:0.5em 0; text-align:center; 
                                  background:#f8fafc; padding:0.5em; border-radius:4px;'>
                            <div>
                                <div style='color:#059669; font-weight:600;'>Entry</div>
                                <div style='color:#1e293b;'>₹{rec['Entry Price']}</div>
                            </div>
                            <div>
                                <div style='color:#dc2626; font-weight:600;'>SL</div>
                                <div style='color:#1e293b;'>₹{rec['Stop Loss']}</div>
                            </div>
                            <div>
                                <div style='color:#0284c7; font-weight:600;'>Target</div>
                                <div style='color:#1e293b;'>₹{rec['Target Price']}</div>
                            </div>
                        </div>
                        <div style='display:flex; justify-content:space-between; margin-top:0.3em;'>
                            <span style='color:#475569; font-size:0.9em;'>{rec['OI Behavior']} | {rec['Volume Strength']}</span>
                            <span style='color:#475569; font-size:0.9em;'>RR: {rr_ratio:.1f}x</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Close grid container
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.info("No high-confidence buy recommendations found in the selected range. Try adjusting the strike range or check the market conditions.")
                for col in ['PCR', 'OI Change %', 'IV Rank', 'Unusual Activity', 'Bid-Ask Spread', 'ITM/ATM/OTM']:
                    if col in df.columns:
                        df_rec[col] = df[col].values
                df_rec['Risk-Reward'] = df_rec.apply(lambda row: (row['Target Price'] - row['Entry Price']) / (row['Entry Price'] - row['Stop Loss']) if (row['Entry Price'] - row['Stop Loss']) != 0 else 0, axis=1)
                def suggest(row):
                    if row['ITM/ATM/OTM'] == 'ATM' and row['Action'] == 'BUY':
                        return 'Straddle/Strangle'
                    elif row['ITM/ATM/OTM'] == 'OTM' and row['Action'] == 'BUY':
                        return 'Directional Option'
                    elif row['ITM/ATM/OTM'] == 'ITM' and row['Action'] == 'BUY':
                        return 'Deep ITM Play'
                    elif row['Action'] == 'AVOID':
                        return 'No Trade'
                    else:
                        return 'Check Manually'
                df_rec['Strategy Suggestion'] = df_rec.apply(suggest, axis=1)
                # Batch Analysis Results Section
                st.markdown("""
                <div style='background: linear-gradient(90deg,#3b82f6,#2563eb); color:white; border-radius:10px; padding:1em; margin:1.5em 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='margin:0; font-size:1.3em; font-weight:600;'>📊 Batch Analysis Results</h3>
                </div>
                """, unsafe_allow_html=True)

                # Calculate metrics first
                very_high_count = len(df_rec[df_rec['Confidence'] == 'Very High'])
                high_count = len(df_rec[df_rec['Confidence'] == 'High'])
                otm_count = len(df_rec[df_rec['ITM/ATM/OTM'] == 'OTM'])
                strong_volume = len(df_rec[df_rec['Volume Strength'] == 'Strong'])
                avg_rr = df_rec[df_rec['Risk-Reward'] > 0]['Risk-Reward'].mean()
                unusual_activity = len(df_rec[df_rec['Unusual Activity'] == 'Yes'])
                
                # Organize metrics into categories
                trade_metrics = {
                    'Trade Opportunities': {
                        'Very High Confidence': very_high_count,
                        'High Confidence': high_count,
                        'Total Opportunities': len(df_rec)
                    },
                    'Position Analysis': {
                        'OTM Options': otm_count,
                        'Strong Volume': strong_volume,
                        'Unusual Activity': unusual_activity
                    },
                    'Risk Metrics': {
                        'Avg Risk-Reward': f'{avg_rr:.2f}' if not pd.isna(avg_rr) else 'N/A',
                        'Max Risk-Reward': f'{df_rec["Risk-Reward"].max():.2f}' if not df_rec["Risk-Reward"].empty else 'N/A',
                        'Median IV': f'{df["IV"].median():.1f}%' if 'IV' in df.columns else 'N/A'
                    }
                }
                
                # Create tabs for different metric categories
                metric_tabs = st.tabs(['💫 Trade Opportunities', '📈 Position Analysis', '⚖️ Risk Metrics'])
                
                for tab, (category, metrics) in zip(metric_tabs, trade_metrics.items()):
                    with tab:
                        cols = st.columns(len(metrics))
                        for col, (label, value) in zip(cols, metrics.items()):
                            with col:
                                st.markdown(f"""
                                <div style='background:white; padding:1em; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.1); margin:0.5em 0;'>
                                    <p style='color:#6b7280; margin:0 0 0.2em 0; font-size:0.9em;'>{label}</p>
                                    <p style='color:#1e293b; margin:0; font-size:1.2em; font-weight:600;'>{value}</p>
                                </div>
                                """, unsafe_allow_html=True)
                
                # Add detailed strategy distribution
                st.markdown("""
                <div style='background: linear-gradient(90deg,#f0f9ff,#e0f2fe); color:#222; border-radius:10px; padding:1em; margin:1em 0;'>
                    <h3 style='margin:0; font-size:1.1em;'>📋 Strategy Distribution</h3>
                </div>
                """, unsafe_allow_html=True)
                
                strategy_cols = st.columns(2)
                
                with strategy_cols[0]:
                    # Confidence distribution
                    confidence_dist = df_rec['Confidence'].value_counts()
                    st.markdown("""
                    <div style='background:white; padding:1em; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.1); margin:0.5em 0;'>
                        <p style='color:#1e293b; margin:0 0 0.5em 0; font-weight:600;'>Confidence Distribution</p>
                    """, unsafe_allow_html=True)
                    
                    for conf, count in confidence_dist.items():
                        color = '#dcfce7' if conf == 'Very High' else '#dbeafe' if conf == 'High' else '#fee2e2'
                        st.markdown(f"""
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:0.3em 0;'>
                            <span style='color:#4b5563;'>{conf}</span>
                            <span style='background:{color}; padding:0.2em 0.6em; border-radius:4px; font-weight:600;'>{count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with strategy_cols[1]:
                    # Strategy suggestions
                    strategy_dist = df_rec['Strategy Suggestion'].value_counts()
                    st.markdown("""
                    <div style='background:white; padding:1em; border-radius:8px; box-shadow:0 1px 2px rgba(0,0,0,0.1); margin:0.5em 0;'>
                        <p style='color:#1e293b; margin:0 0 0.5em 0; font-weight:600;'>Recommended Strategies</p>
                    """, unsafe_allow_html=True)
                    
                    for strategy, count in strategy_dist.items():
                        st.markdown(f"""
                        <div style='display:flex; justify-content:space-between; align-items:center; padding:0.3em 0;'>
                            <span style='color:#4b5563;'>{strategy}</span>
                            <span style='background:#f1f5f9; padding:0.2em 0.6em; border-radius:4px; font-weight:600;'>{count}</span>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                # Organize recommendations by confidence and action
                very_high_buys = df_rec[(df_rec['Action'] == 'BUY') & (df_rec['Confidence'] == 'Very High')]
                high_buys = df_rec[(df_rec['Action'] == 'BUY') & (df_rec['Confidence'] == 'High')]
                moderate_buys = df_rec[(df_rec['Action'] == 'BUY') & (df_rec['Confidence'] == 'Moderate')]
                avoid_trades = df_rec[df_rec['Action'] == 'AVOID']

                # Show trading analysis
                st.subheader("📈 High Probability Trades")
                
                # Convert recommendations list to DataFrame if not already
                if isinstance(recommendations, list):
                    recommendations_df = pd.DataFrame(recommendations)
                else:
                    recommendations_df = recommendations
                
                # Get high and very high confidence buy recommendations
                very_high_buys = recommendations_df[
                    (recommendations_df['Confidence'] == 'Very High') & 
                    (recommendations_df['Action'] == 'BUY')
                ]
                high_buys = recommendations_df[
                    (recommendations_df['Confidence'] == 'High') & 
                    (recommendations_df['Action'] == 'BUY')
                ]

                if len(very_high_buys) > 0 or len(high_buys) > 0:
                    combined_high_prob = pd.concat([very_high_buys, high_buys]) if not very_high_buys.empty and not high_buys.empty else (very_high_buys if not very_high_buys.empty else high_buys)
                    
                    # Calculate Risk-Reward ratio
                    combined_high_prob['Risk-Reward'] = combined_high_prob.apply(
                        lambda row: (row['Target Price'] - row['Entry Price']) / (row['Entry Price'] - row['Stop Loss']) 
                        if (row['Entry Price'] - row['Stop Loss']) != 0 else 0, 
                        axis=1
                    )
                    
                    # Keep only essential columns for clarity
                    display_cols = ['Option Type', 'Strike Price', 'Entry Price', 'Target Price', 'Stop Loss', 
                                    'Confidence', 'Volume Strength', 'Risk-Reward']
                    display_df = combined_high_prob[display_cols].copy()
                    
                    def apply_style(row):
                        color = '#dcfce7' if row['Confidence'] == 'Very High' else '#dbeafe'  # Light green for very high, light blue for high
                        return [f'background-color: {color}'] * len(row)
                    
                    styled_df = display_df.style.apply(apply_style, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    st.info("💡 Trading Opportunity Details:\n"
                           "- 🟢 Green rows indicate very high confidence trades (90%+ probability)\n"
                           "- 🟡 Yellow rows indicate high confidence trades (80-89% probability)\n"
                           "- Entry Price: Current market price for immediate entry\n"
                           "- Target & Stop Loss: Suggested exit points based on technical analysis\n"
                           "- Risk-Reward: Potential profit vs. potential loss ratio")
                else:
                    st.info("No high probability trades found at current market conditions. "
                           "Consider waiting for better opportunities or analyzing different strike ranges.")

                # Only show specialized analysis if we have high-volume OTM opportunities
                otm_volume = df_rec[(df_rec['ITM/ATM/OTM'] == 'OTM') & 
                                   (df_rec['Action'] == 'BUY') & 
                                   (df_rec['Confidence'].isin(['High', 'Very High'])) & 
                                   (df_rec['Volume Strength'] == 'Strong')].copy()
                
                if not otm_volume.empty:
                    # Calculate Risk-Reward ratio for OTM opportunities
                    otm_volume['Risk-Reward'] = otm_volume.apply(
                        lambda row: (row['Target Price'] - row['Entry Price']) / (row['Entry Price'] - row['Stop Loss']) 
                        if (row['Entry Price'] - row['Stop Loss']) != 0 else 0, 
                        axis=1
                    )
                    
                    st.markdown("""
                    <div style='background: linear-gradient(90deg,#f8fafc,#f1f5f9); color:#222; border-radius:10px; padding:1em; margin:1em 0;'>
                        <h3 style='margin:0; font-size:1.2em;'>🎯 OTM Trading Opportunities</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("""
                    <div style='background:#f0f9ff; padding:0.5em; border-radius:8px; margin-bottom:1em;'>
                        <p style='color:#075985; margin:0;'>High-probability out-of-the-money options with strong volume</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    display_cols = ['Option Type', 'Strike Price', 'Entry Price', 'Target Price', 'Stop Loss', 
                                    'Confidence', 'Volume Strength', 'Risk-Reward']
                    otm_display = otm_volume[display_cols]
                    styled_df = otm_display.style.background_gradient(subset=['Risk-Reward'], cmap='RdYlGn')
                    st.dataframe(styled_df, height=200)

                # OTM Opportunities
                otm_volume = df_rec[(df_rec['ITM/ATM/OTM'] == 'OTM') & 
                                   (df_rec['Action'] == 'BUY') & 
                                   (df_rec['Confidence'].isin(['High', 'Very High'])) & 
                                   (df_rec['Volume Strength'] == 'Strong')]
                if not otm_volume.empty:
                    st.markdown("#### 🎯 Premium OTM Opportunities")
                    st.markdown("""
                    <div style='background:#f0f9ff; padding:0.5em; border-radius:8px; margin-bottom:1em;'>
                        <p style='color:#075985; margin:0;'>Out-of-the-money options with high potential returns and confirmed volume</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(otm_volume.style.background_gradient(subset=['Risk-Reward'], cmap='RdYlGn'), height=200)
                st.download_button("⬇️ Download Recommendations as CSV", df_rec.to_csv(index=False), "recommendations.csv")

def display_high_probability_trades(recommendations_df):
    if recommendations_df is None or recommendations_df.empty:
        st.warning("No high probability trades found in the current data.")
        return

    # Ensure all required columns exist
    required_cols = ['Strike Price', 'Option Type', 'Action', 'Confidence Label', 'LTP']
    if not all(col in recommendations_df.columns for col in required_cols):
        st.error("Required columns missing from recommendations data.")
        return

    # Filter for only BUY recommendations with Very High confidence
    high_prob_trades = recommendations_df[
        (recommendations_df['Action'] == 'BUY') & 
        (recommendations_df['Confidence Label'] == 'Very High')
    ].copy()

    if high_prob_trades.empty:
        st.info("No high probability trades found with current filters.")
        return

    # Create display columns
    display_cols = ['Strike Price', 'Option Type', 'LTP']
    if 'IV' in high_prob_trades.columns:
        display_cols.append('IV')
    if 'Volume' in high_prob_trades.columns:
        display_cols.append('Volume')
    display_cols.extend(['OI Change %', 'PCR', 'Action', 'Confidence Label'])

    # Format and display the trades
    with st.container():
        st.subheader("🎯 High Probability Trades")
        st.write("These are the highest confidence BUY signals based on our analysis:")
        
        # Create a grid of cards for the trades
        cols = st.columns(3)
        for idx, row in high_prob_trades.iterrows():
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"""
                    <div style="border:1px solid #4CAF50; border-radius:10px; padding:10px; margin:5px;">
                        <h3 style="color:#4CAF50; margin:0;">{row['Strike Price']} {row['Option Type']}</h3>
                        <p style="margin:5px 0;">💰 LTP: {row['LTP']:.2f}</p>
                        {'<p style="margin:5px 0;">📊 IV: ' + f"{row['IV']:.2f}%" + '</p>' if 'IV' in row else ''}
                        {'<p style="margin:5px 0;">📈 Volume: ' + f"{row['Volume']:,.0f}" + '</p>' if 'Volume' in row else ''}
                        <p style="margin:5px 0;">🔄 OI Change: {row['OI Change %']:.1f}%</p>
                        <p style="margin:5px 0;">⚖️ PCR: {row['PCR']:.2f}</p>
                        <p style="color:#4CAF50; font-weight:bold; margin:5px 0;">{row['Action']} ({row['Confidence Label']})</p>
                    </div>
                    """, unsafe_allow_html=True)