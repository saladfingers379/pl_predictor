import pandas as pd
from src.backtest import Backtester
from src.feature_engineering import prepare_training_data

def verify_backtest():
    print("Loading data...")
    # Load a sample of data (assuming the file exists from previous context, otherwise I'll need to find it)
    # I'll check the data directory first.
    try:
        df = pd.read_csv('data/E0.csv') # Assuming standard football-data.co.uk format
    except FileNotFoundError:
        print("Data file not found in data/E0.csv. Checking other locations...")
        # Fallback or mock data if needed, but let's try to find real data first.
        import os
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.endswith('.csv') and 'E0' in file:
                    df = pd.read_csv(os.path.join(root, file))
                    print(f"Found data at {os.path.join(root, file)}")
                    break
            else:
                continue
            break
        else:
            print("No suitable CSV found. Creating mock data.")
            # Create mock data
            dates = pd.date_range(start='2020-08-01', periods=200, freq='W')
            data = {
                'Date': dates,
                'HomeTeam': ['TeamA', 'TeamB'] * 100,
                'AwayTeam': ['TeamB', 'TeamA'] * 100,
                'FTHG': [1, 2] * 100,
                'FTAG': [2, 1] * 100,
                'FTR': ['A', 'H', 'D'] * 66 + ['H', 'A'],
                'HS': [10] * 200, 'AS': [10] * 200, 'HST': [5] * 200, 'AST': [5] * 200,
                'B365H': [2.0] * 200, 'B365D': [3.0] * 200, 'B365A': [4.0] * 200
            }
            df = pd.DataFrame(data)

    print("Data loaded. Columns:", df.columns.tolist())
    
    # Define features
    features = [
        'Home_Rolling_GF', 'Away_Rolling_GF',
        'Home_Rolling_GA', 'Away_Rolling_GA',
        'Home_ELO', 'Away_ELO',
        'Home_RestDays', 'Away_RestDays',
        'Home_HomeForm_GF', 'Away_AwayForm_GF'
    ]
    
    print("Initializing Backtester...")
    backtester = Backtester(initial_bankroll=1000, staking_method='kelly', kelly_fraction=0.1)
    
    print("Running Backtest...")
    # Run on a small window to verify it works
    # We need enough data for rolling stats (window=5) and initial training
    # Let's try to backtest the last 20% of the data
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True) # football-data usually dd/mm/yy
    
    # Sort
    df = df.sort_values('Date')
    
    # Pick a start season/date that allows for training
    # Just use the last year in the data
    last_date = df['Date'].max()
    start_year = last_date.year
    if last_date.month < 8:
        start_year -= 1
        
    print(f"Backtesting starting from season {start_year}-{start_year+1}")
    
    backtester.run(df, features, start_season=f"{start_year}-{start_year+1}", retrain_every=10)
    
    print("Verification complete.")

if __name__ == "__main__":
    verify_backtest()
