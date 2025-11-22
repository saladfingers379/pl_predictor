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
            'Away_Rolling_GF', 'Away_Rolling_GA', 'Away_Rolling_Pts'
        ]
        
        print(f"Training XGBoost model with features: {features}")
        predictor = XGBoostPredictor(features=features)
        predictor.train(train_df)
        predictor.get_feature_importance()
        
    elif args.predict:
        print("Running prediction...")
        from src.predict import predict_upcoming_fixtures
        # Check if user wants to predict all future games
        # We need to add the argument to parser first
        predict_upcoming_fixtures(use_full_schedule=args.all)
    elif args.backtest:
        print("Running backtest...")
        df = load_historical_data('Historical_data_from_football_dot_co_dot_uk')
        
        # Define features to use (same as training)
        features = [
            'Home_Rolling_GF', 'Home_Rolling_GA', 'Home_Rolling_Pts',
            'Away_Rolling_GF', 'Away_Rolling_GA', 'Away_Rolling_Pts'
        ]
        
        from src.backtest import Backtester
        backtester = Backtester()
        backtester.run(df, features)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
