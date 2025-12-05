import pandas as pd
from src.models import XGBoostPredictor, ElasticNetPredictor, EnsemblePredictor
from src.feature_engineering import prepare_training_data
from src.data_loader import load_historical_data

def standardize_team_names(df):
    """
    Standardizes team names to match historical data format.
    Maps full/official names from APIs to the shortened names used in historical data.
    """
    name_map = {
        'Brighton and Hove Albion': 'Brighton',
        'Manchester City': 'Man City',
        'Manchester United': 'Man United',
        'Newcastle United': 'Newcastle',
        'Nottingham Forest': "Nott'm Forest",
        'Tottenham Hotspur': 'Tottenham',
        'West Ham United': 'West Ham',
        'Wolverhampton Wanderers': 'Wolves',
        'Leeds United': 'Leeds'
    }

    # Apply mapping to both Home and Away teams
    if 'HomeTeam' in df.columns:
        df['HomeTeam'] = df['HomeTeam'].replace(name_map)
    if 'AwayTeam' in df.columns:
        df['AwayTeam'] = df['AwayTeam'].replace(name_map)

    return df

def predict_upcoming_fixtures(historical_data_path='Historical_data_from_football_dot_co_dot_uk',
                              fixtures_path='data/latest_odds.csv',
                              use_full_schedule=False,
                              model_type='ensemble'):
    """
    Trains the model on all historical data and predicts upcoming fixtures.
    """
    # 1. Load Historical Data
    print("Loading historical data...")
    hist_df = load_historical_data(historical_data_path)
    
    # 2. Load Upcoming Fixtures
    if use_full_schedule:
        fixtures_path = 'data/fixtures.csv'
        print(f"Loading FULL schedule from {fixtures_path}...")
    else:
        print(f"Loading upcoming fixtures from {fixtures_path}...")
        
    try:
        upcoming_df = pd.read_csv(fixtures_path)
    except FileNotFoundError:
        print(f"Error: {fixtures_path} not found. Run 'python src/fixture_fetcher.py' first.")
        return

    # Ensure upcoming fixtures have necessary columns for feature engineering
    # We need Date, HomeTeam, AwayTeam. 
    # The Odds API returns: Date, HomeTeam, AwayTeam, HomeOdds, DrawOdds, AwayOdds
    # We need to ensure column names match what our pipeline expects
    
    # Convert Date to datetime and ensure timezone naive (UTC)
    upcoming_df['Date'] = pd.to_datetime(upcoming_df['Date']).dt.tz_convert(None)

    # Standardize team names to match historical data
    upcoming_df = standardize_team_names(upcoming_df)
    
    # Filter for future games only
    now = pd.Timestamp.now()
    upcoming_df = upcoming_df[upcoming_df['Date'] > now].copy()
    
    if upcoming_df.empty:
        print("No future fixtures found.")
        return
        
    print(f"Found {len(upcoming_df)} future matches to predict.")
    
    # Ensure historical dates are also timezone naive
    if hist_df['Date'].dt.tz is not None:
        hist_df['Date'] = hist_df['Date'].dt.tz_convert(None)
            
    # 3. Mark upcoming fixtures before combining
    # This allows us to identify them later even if dates collide with historical data
    upcoming_df['IsFutureFixture'] = True

    # Combine Dataframes
    # pd.concat handles alignment and missing columns automatically
    combined_df = pd.concat([hist_df, upcoming_df], ignore_index=True)
    # Sort by date, then by IsFutureFixture to ensure consistent ordering
    # Historical matches (IsFutureFixture=NaN) will come before future fixtures with same date
    combined_df['IsFutureFixture'] = combined_df['IsFutureFixture'].fillna(False)
    combined_df = combined_df.sort_values(['Date', 'IsFutureFixture']).reset_index(drop=True)
    
    # 4. Prepare Data (Calculate Features)
    print("Calculating features...")
    # is_training=False allows rows with NaN Target (our upcoming games) to be kept
    processed_df = prepare_training_data(combined_df, is_training=False)
    
    # 5. Separate Training and Prediction Sets
    # Training set: Rows where we have a Target (result)
    train_df = processed_df.dropna(subset=['Target'])

    # Prediction set: Rows marked as future fixtures (prevents date collisions with historical data)
    predict_df = processed_df[processed_df['IsFutureFixture'] == True].copy()
    
    if predict_df.empty:
        print("No upcoming fixtures found in processed data (check date parsing).")
        return

    # 6. Train Model
    print(f"Training {model_type} model on {len(train_df)} historical matches...")
    # CRITICAL: These features must match the features used in main.py training
    features = [
        'Home_Rolling_GF', 'Home_Rolling_GA', 'Home_Rolling_Pts',
        'Away_Rolling_GF', 'Away_Rolling_GA', 'Away_Rolling_Pts',
        'Home_ELO', 'Away_ELO',
        'Home_RestDays', 'Away_RestDays',
        'Home_HomeForm_GF', 'Away_AwayForm_GF'
    ]

    # Select model based on type
    if model_type == 'xgboost':
        predictor = XGBoostPredictor(features=features)
    elif model_type == 'elasticnet':
        predictor = ElasticNetPredictor(features=features)
    elif model_type == 'ensemble':
        predictor = EnsemblePredictor(features=features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    predictor.train(train_df)
    
    # 7. Predict
    print(f"Predicting {len(predict_df)} upcoming matches...")
    probs = predictor.predict_proba(predict_df)
    
    # 8. Format Output
    results = predict_df[['Date', 'HomeTeam', 'AwayTeam']].copy()
    results['Prob_Home'] = probs[:, 0]
    results['Prob_Draw'] = probs[:, 1]
    results['Prob_Away'] = probs[:, 2]
    
    # Calculate Fair Odds (1 / Probability)
    results['FairOdds_Home'] = 1 / results['Prob_Home']
    results['FairOdds_Draw'] = 1 / results['Prob_Draw']
    results['FairOdds_Away'] = 1 / results['Prob_Away']
    
    # Display
    print("\n--- Upcoming Fixture Predictions ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_rows', 20) # Don't print 300 rows
    
    # Format for nice printing
    display_cols = ['Date', 'HomeTeam', 'AwayTeam', 'FairOdds_Home', 'FairOdds_Draw', 'FairOdds_Away']
    print(results[display_cols].head(20).to_string(index=False, formatters={
        'FairOdds_Home': '{:.2f}'.format,
        'FairOdds_Draw': '{:.2f}'.format,
        'FairOdds_Away': '{:.2f}'.format
    }))
    
    if len(results) > 20:
        print(f"\n... and {len(results)-20} more matches.")
    
    # Save to CSV
    output_filename = 'data/future_predictions.csv' if use_full_schedule else 'data/predictions.csv'
    results.to_csv(output_filename, index=False)
    print(f"\nPredictions saved to {output_filename}")

if __name__ == "__main__":
    predict_upcoming_fixtures()
