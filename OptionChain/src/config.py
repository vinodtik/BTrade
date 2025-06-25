# Project configuration and constants

# Minimum OI, volume, and other thresholds for filtering
MIN_OI = 100  # Reduced minimum OI threshold
MIN_VOLUME = 50  # Reduced minimum volume threshold
MAX_BID_ASK_SPREAD = 5.0  # Increased max bid-ask spread
PROXIMITY_TO_SPOT = 1000  # Increased range around spot price
RISK_REWARD_RATIO = 2.0

# Columns expected in the CSV
CSV_COLUMNS = [
    'Strike Price', 'Option Type', 'LTP', 'Bid', 'Ask', 'OI', 'Change in OI',
    'Volume', 'IV', 'Spot Price'
]
