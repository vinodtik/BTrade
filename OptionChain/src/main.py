import pandas as pd
from loader import load_option_chain_csv
from strategy import filter_liquid_options, filter_proximity_to_spot
from recommender import recommend_trades
from report import print_recommendations_table

def main(csv_path):
    df = load_option_chain_csv(csv_path)
    df = filter_liquid_options(df)
    df = filter_proximity_to_spot(df)
    recommendations = recommend_trades(df)
    print_recommendations_table(recommendations)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python main.py <option_chain_csv>")
    else:
        main(sys.argv[1])
