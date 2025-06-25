from tabulate import tabulate
import pandas as pd

def print_recommendations_table(recommendations):
    df = pd.DataFrame(recommendations)
    # Highlight best for scalp (highest volume, BUY action, near spot)
    scalp = df[(df['Action'] == 'BUY')].sort_values(['Volume Strength', 'Entry Price'], ascending=[False, True]).head(2)
    # Highlight best for swing (highest target, BUY action)
    swing = df[(df['Action'] == 'BUY')].sort_values('Target Price', ascending=False).head(1)
    print("\n=== All Recommendations ===")
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    print("\n🔥 Best 1–2 strike prices for intraday scalp:")
    print(tabulate(scalp, headers='keys', tablefmt='psql', showindex=False))
    print("\n📈 Best 1 for swing or breakout move:")
    print(tabulate(swing, headers='keys', tablefmt='psql', showindex=False))
