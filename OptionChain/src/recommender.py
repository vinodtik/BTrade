import math
import pandas as pd
from analyzer import analyze_oi_behavior, analyze_volume_strength
from strategy import filter_liquid_options, filter_proximity_to_spot
from greeks import calculate_greeks
from datetime import datetime

# Risk-Reward calculation
RISK_REWARD_RATIO = 2.0
RISK_PERCENT = 20  # 20% stop loss by default

# Default risk-free rate and expiry days (can be improved to read from config or UI)
RISK_FREE_RATE = 0.06
DAYS_TO_EXPIRY = 2  # fallback if not available

# --- Minimum absolute target for OTM options ---
MIN_TARGET_ABS = 5  # You can adjust this value as needed

# Confidence level mapping
CONFIDENCE_LABELS = {
    1: 'Very Low',
    2: 'Low',
    3: 'Moderate',
    4: 'High',
    5: 'Very High'
}

def recommend_trades(df):
    """
    Generates trade recommendations for each valid option.
    Simple, realistic: Entry = LTP, Stop Loss = 20% below entry, Target = entry + (entry - stop_loss) * 2
    """
    results = []
    avg_volume = df['Volume'].mean()
    spot = df['Spot Price'].iloc[0]
    expiry_date = None
    today = datetime.now().date()
    if 'Expiry' in df.columns:
        expiry_date = pd.to_datetime(df['Expiry'].iloc[0]).date()
        days_to_expiry = max((expiry_date - today).days, 1)
    else:
        days_to_expiry = DAYS_TO_EXPIRY
    T = days_to_expiry / 365
    for option_type in ['CALL', 'PUT']:
        df_type = df[df['Option Type'] == option_type]
        avg_ltp = df_type['LTP'].mean() if not df_type.empty else 0
        for _, row in df_type.iterrows():
            oi_behavior = analyze_oi_behavior(row, avg_ltp)
            volume_strength = analyze_volume_strength(row, avg_volume)
            ltp = float(row['LTP']) if row['LTP'] > 0 else float(row['Bid'])
            if ltp <= 0:
                continue  # skip invalid
            entry = round(ltp, 2)
            stop_loss = round(entry * (1 - RISK_PERCENT / 100), 2)
            target = round(entry + ((entry - stop_loss) * RISK_REWARD_RATIO), 2)
            K = row['Strike Price']
            greeks = calculate_greeks(option_type, spot, K, T, RISK_FREE_RATE, row['IV'] / 100 if 'IV' in row and row['IV'] > 0 else 0.18)
            delta = greeks['Delta']
            moneyness = abs(K - spot) / spot
            # More flexible action determination
            action = 'AVOID'
            if (oi_behavior in ['Long Build-up', 'Short Covering'] and volume_strength in ['Strong', 'Moderate']) or \
               (oi_behavior == 'Long Build-up' and row['Change in OI'] > 0) or \
               (volume_strength == 'Strong' and row['Volume'] > avg_volume * 1.5):
                action = 'BUY'
            
            # Build detailed reason
            reason = []
            if action == 'BUY':
                if abs(K - spot) < 100:  # Increased range for near-spot options
                    reason.append('Within reasonable range of spot price')
                if oi_behavior == 'Long Build-up':
                    reason.append('Fresh long positions building up')
                elif oi_behavior == 'Short Covering':
                    reason.append('Short covering indicates potential upside')
                if volume_strength == 'Strong':
                    reason.append('High trading activity')
                elif volume_strength == 'Moderate':
                    reason.append('Decent trading volume')
                if row['Change in OI'] > 0:
                    reason.append('Positive OI change')
                if row['Volume'] > avg_volume:
                    reason.append('Above average volume')
                reason.append(f"Risk:Reward = 1:{RISK_REWARD_RATIO}")
            else:
                if volume_strength == 'Weak':
                    reason.append('Low trading volume')
                if oi_behavior not in ['Long Build-up', 'Short Covering']:
                    reason.append('No bullish OI pattern')
            # --- Enhanced Confidence Level Calculation ---
            confidence = 0
            if action == 'BUY':
                # OI-based confidence
                if oi_behavior == 'Long Build-up':
                    confidence += 2
                elif oi_behavior == 'Short Covering':
                    confidence += 1.5
                
                # Volume-based confidence
                if volume_strength == 'Strong':
                    confidence += 2
                elif volume_strength == 'Moderate':
                    confidence += 1
                
                # Greek-based confidence
                if 0.30 < abs(delta) < 0.70:  # Widened delta range
                    confidence += 1
                if abs(greeks['Theta']) < 0.03:  # Relaxed theta threshold
                    confidence += 1
                if abs(greeks['Vega']) > 0.04:
                    confidence += 1
                
                # Additional factors
                if row['Volume'] > avg_volume * 1.5:
                    confidence += 1
                if row['Change in OI'] > 0:
                    confidence += 0.5
                
                # Moneyness impact
                if moneyness < 0.03:  # Near ATM
                    confidence += 1
                elif moneyness > 0.10:  # Deep OTM/ITM
                    confidence -= 0.5
                
                confidence = max(1, min(confidence, 5))  # Scale to 1-5
            else:
                confidence = 1  # Base confidence for AVOID recommendations
            
            # Round the confidence score and ensure it's within valid range
            confidence_key = int(round(confidence))
            confidence_key = max(1, min(confidence_key, 5))
            confidence_label = CONFIDENCE_LABELS[confidence_key]
            
            results.append({
                'Option Type': option_type,
                'Strike Price': K,
                'Action': action,
                'Entry Price': entry,
                'Stop Loss': stop_loss,
                'Target Price': target,
                'Delta': delta,
                'Gamma': greeks['Gamma'],
                'Theta': greeks['Theta'],
                'Vega': greeks['Vega'],
                'Rho': greeks['Rho'],
                'OI Behavior': oi_behavior,
                'Volume Strength': volume_strength,
                'Reason': ' '.join(reason),
                'Confidence': confidence_label
            })
    return results

def calculate_recommendation(row):
    features = {
        'IV': row.get('IV', 0),  # Default to 0 if IV doesn't exist
        'PCR': row.get('PCR', 0),
        'OI Change %': row.get('OI Change %', 0),
        'Unusual Activity': row.get('Unusual Activity', 'No'),
        'Moneyness': row.get('ITM/ATM/OTM', 'OTM')
    }
    
    confidence_score = 0
    action = "HOLD"  # Default action
    
    # Calculate base confidence from IV and PCR
    if 'IV' in row:
        iv_rank = row.get('IV Rank', 0)
        confidence_score += iv_rank * 20  # Weight IV contribution to confidence
    
    # Add PCR contribution
    pcr_score = 0
    if features['PCR'] > 1.5:
        pcr_score = 20
    elif features['PCR'] < 0.67:
        pcr_score = -20
    confidence_score += pcr_score

    # Add OI Change contribution
    oi_change = features['OI Change %']
    if abs(oi_change) > 50:
        confidence_score += (20 if oi_change > 0 else -20)
    elif abs(oi_change) > 25:
        confidence_score += (10 if oi_change > 0 else -10)
    
    # Add Unusual Activity bonus
    if features['Unusual Activity'] == 'Yes':
        confidence_score += 10
    
    # Moneyness multiplier
    moneyness_multiplier = {
        'ITM': 1.2,
        'ATM': 1.1,
        'OTM': 0.9
    }.get(features['Moneyness'], 1.0)
    
    confidence_score *= moneyness_multiplier
    
    # Normalize confidence score to 0-100 range
    confidence_score = max(0, min(100, confidence_score + 50))
    
    # Determine action based on features and confidence
    if confidence_score >= 70:
        if oi_change > 0 or features['PCR'] < 0.67:
            action = "BUY"
        elif oi_change < 0 or features['PCR'] > 1.5:
            action = "SELL"
    elif confidence_score <= 30:
        if oi_change < 0 or features['PCR'] > 1.5:
            action = "BUY"
        elif oi_change > 0 or features['PCR'] < 0.67:
            action = "SELL"
    
    # Map confidence score to label
    confidence_label = None
    score = int(confidence_score)
    if score >= 90:
        confidence_label = "Very High"
    elif score >= 70:
        confidence_label = "High"
    elif score >= 50:
        confidence_label = "Medium"
    elif score >= 30:
        confidence_label = "Low"
    else:
        confidence_label = "Very Low"
    
    return {
        'Action': action,
        'Confidence': score,
        'Confidence Label': confidence_label,
        'Recommendation Reasons': []  # We'll populate this later
    }
