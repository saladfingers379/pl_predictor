import argparse
import pandas as pd
from src.data_loader import load_historical_data
from src.feature_engineering import prepare_training_data
from src.models import XGBoostPredictor

def main():
    parser = argparse.ArgumentParser(description="Premier League Match Odds Predictor")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict upcoming fixtures')
    parser.add_argument('--all', action='store_true', help='Predict ALL future fixtures (from full schedule)')
    parser.add_argument('--backtest', action='store_true', help='Run backtest on historical data')
    parser.add_argument('--season', type=str, default='2023-2024', help='Season to backtest (e.g., 2023-2024, 23/24, or 2022-2023)')
    parser.add_argument('--retrain-every', type=int, default=50, help='Number of matches to wait before retraining the model during backtest')
    
    args = parser.parse_args()
    
    if args.train:
        print("Loading historical data...")
        df = load_historical_data('Historical_data_from_football_dot_co_dot_uk')
        
        print("Preparing training data...")
        train_df = prepare_training_data(df)
        print(f"Training data shape: {train_df.shape}")
        
        # Define features to use
        features = [
            'Home_Rolling_GF', 'Home_Rolling_GA', 'Home_Rolling_Pts',
            'Away_Rolling_GF', 'Away_Rolling_GA', 'Away_Rolling_Pts',
            'Home_ELO', 'Away_ELO',
            'Home_RestDays', 'Away_RestDays',
            'Home_HomeForm_GF', 'Away_AwayForm_GF'
        ]
        
        print(f"Training XGBoost model with features: {features}")
        predictor = XGBoostPredictor(features=features)
        predictor.train(train_df, tune=True) # Enable tuning for manual training
        predictor.get_feature_importance()
        
        # Export Current ELO
        print("Exporting current ELO ratings...")
        # Get the last row for each team to get their latest ELO
        # We need to look at both Home and Away columns
        
        # Create a long format of team, date, elo
        home_elo = train_df[['Date', 'HomeTeam', 'Home_ELO']].rename(columns={'HomeTeam': 'Team', 'Home_ELO': 'ELO'})
        away_elo = train_df[['Date', 'AwayTeam', 'Away_ELO']].rename(columns={'AwayTeam': 'Team', 'Away_ELO': 'ELO'})
        
        all_elo = pd.concat([home_elo, away_elo])
        latest_elo = all_elo.sort_values('Date').groupby('Team').last().reset_index()
        latest_elo = latest_elo[['Team', 'ELO']].sort_values('ELO', ascending=False)
        
        latest_elo.to_csv('data/current_elo.csv', index=False)
        print(f"Current ELO ratings saved to data/current_elo.csv")
        print(latest_elo.head(10))
        
    elif args.predict:
        print("Running prediction...")
        from src.predict import predict_upcoming_fixtures
        # Check if user wants to predict all future games
        # We need to add the argument to parser first
        predict_upcoming_fixtures(use_full_schedule=args.all)
    elif args.backtest:
        print(f"Running backtest for season {args.season}...")
        df = load_historical_data('Historical_data_from_football_dot_co_dot_uk')

        # Define features to use (same as training)
        features = [
            'Home_Rolling_GF', 'Home_Rolling_GA', 'Home_Rolling_Pts',
            'Away_Rolling_GF', 'Away_Rolling_GA', 'Away_Rolling_Pts',
            'Home_ELO', 'Away_ELO',
            'Home_RestDays', 'Away_RestDays',
            'Home_HomeForm_GF', 'Away_AwayForm_GF'
        ]

        from src.backtest import Backtester
        backtester = Backtester(initial_bankroll=1000, staking_method='kelly')
        backtester.run(df, features, start_season=args.season, retrain_every=args.retrain_every)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
