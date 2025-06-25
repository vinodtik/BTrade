import pandas as pd
from config import MIN_OI, MIN_VOLUME, MAX_BID_ASK_SPREAD, PROXIMITY_TO_SPOT

def filter_liquid_options(df):
    """
    Pass-through function that no longer filters the data.
    Only logs the data statistics.
    """
    if df.empty:
        return df
        
    total_rows = len(df)
    print(f"Data statistics:")
    print(f"Total rows: {total_rows}")
    print(f"Unique strikes: {df['Strike Price'].nunique()}")
    
    return df

def filter_proximity_to_spot(df):
    """
    Pass-through function that no longer filters the data.
    Only logs the spot price information.
    """
    if df.empty:
        return df
        
    spot = df['Spot Price'].iloc[0]
    print(f"Spot price: {spot}")
    print(f"Strike range: {df['Strike Price'].min()} to {df['Strike Price'].max()}")
    
    return df
