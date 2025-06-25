import pandas as pd

def analyze_oi_behavior(row, avg_ltp):
    """
    Determines OI behavior: Long Build-up, Short Covering, etc.
    """
    if row['Change in OI'] > 0 and row['LTP'] > avg_ltp:
        return 'Long Build-up'
    elif row['Change in OI'] < 0 and row['LTP'] > avg_ltp:
        return 'Short Covering'
    elif row['Change in OI'] > 0 and row['LTP'] < avg_ltp:
        return 'Short Build-up'
    elif row['Change in OI'] < 0 and row['LTP'] < avg_ltp:
        return 'Long Unwinding'
    else:
        return 'Neutral'

def analyze_volume_strength(row, avg_volume):
    if row['Volume'] > avg_volume * 1.2:
        return 'Strong'
    elif row['Volume'] < avg_volume * 0.8:
        return 'Weak'
    else:
        return 'Normal'
