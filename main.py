import argparse
import pandas as pd
from src.data_loader import load_historical_data
from src.feature_engineering import prepare_training_data
from src.models import XGBoostPredictor, ElasticNetPredictor, EnsemblePredictor

def main():
    parser = argparse.ArgumentParser(description="Premier League Match Odds Predictor")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Predict upcoming fixtures')
    parser.add_argument('--all', action='store_true', help='Predict ALL future fixtures (from full schedule)')
    parser.add_argument('--backtest', action='store_true', help='Run backtest on historical data')
    parser.add_argument('--season', type=str, default='2023-2024', help='Season(s) to backtest: single (2023-2024), multiple (2022-2023,2023-2024), or "all"')
    parser.add_argument('--retrain-every', type=int, default=50, help='Number of matches to wait before retraining the model during backtest')
    parser.add_argument('--model', type=str, default='xgboost', choices=['xgboost', 'elasticnet', 'ensemble'], help='Model to use: xgboost, elasticnet, or ensemble (default: xgboost)')
    parser.add_argument('--max-odds', type=float, default=2.0, help='Maximum odds to bet on (default: 2.0 - favorites only)')
    parser.add_argument('--list-seasons', action='store_true', help='List available seasons in the data and exit')
    parser.add_argument('--update-data', action='store_true', help='Fetch latest odds and schedule from APIs')

    args = parser.parse_args()

    # Handle --update-data
    if args.update_data:
        print("Updating data from APIs...")
        from src.fixture_fetcher import get_upcoming_fixtures, scrape_full_schedule
        import os
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        print("Fetching upcoming fixtures from The Odds API...")
        try:
            df_api = get_upcoming_fixtures()
            if not df_api.empty:
                output_path = os.path.join('data', 'latest_odds.csv')
                df_api.to_csv(output_path, index=False)
                print(f"Saved odds to {output_path}")
                print(df_api.head())
            else:
                print("No odds returned (check API key or season status).")
        except Exception as e:
            print(f"Error fetching odds: {e}")
            
        print("\nFetching full schedule from fixturedownload.com...")
        try:
            df_schedule = scrape_full_schedule()
            if not df_schedule.empty:
                output_path = os.path.join('data', 'fixtures.csv')
                df_schedule.to_csv(output_path, index=False)
                print(f"Saved full schedule to {output_path}")
            else:
                print("No schedule returned.")
        except Exception as e:
            print(f"Error fetching schedule: {e}")
        
        return

    # Handle --list-seasons first
    if args.list_seasons:
        print("Loading historical data...")
        df = load_historical_data('Historical_data_from_football_dot_co_dot_uk')
        df['Date'] = pd.to_datetime(df['Date'])

        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month

        print("\n" + "=" * 70)
        print("AVAILABLE SEASONS IN DATA")
        print("=" * 70)

        seasons_info = []
        for year in sorted(df['Year'].unique()):
            year = int(year)  # Convert to int to avoid float formatting issues
            first_half = df[(df['Year'] == year) & (df['Month'] >= 8)]
            second_half = df[(df['Year'] == year + 1) & (df['Month'] <= 7)]

            total_matches = len(first_half) + len(second_half)
            if total_matches > 100:  # Reasonable threshold for a complete season
                season_str = f"{year}-{year + 1}"
                date_range = f"{first_half['Date'].min().strftime('%Y-%m-%d')} to {second_half['Date'].max().strftime('%Y-%m-%d') if len(second_half) > 0 else 'N/A'}"
                complete = "✓" if len(second_half) > 50 else "✗ (incomplete)"
                seasons_info.append((season_str, total_matches, date_range, complete))

        print(f"\n{'Season':<15} {'Matches':<10} {'Date Range':<35} {'Status':<15}")
        print("-" * 70)
        for season, matches, date_range, status in seasons_info:
            print(f"{season:<15} {matches:<10} {date_range:<35} {status:<15}")

        print("\n" + "=" * 70)
        print(f"Total complete seasons: {sum(1 for s in seasons_info if '✓' in s[3])}")
        print("=" * 70)
        return
    
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
        
        # Select model based on argument
        if args.model == 'xgboost':
            print(f"Training XGBoost model with features: {features}")
            predictor = XGBoostPredictor(features=features)
        elif args.model == 'elasticnet':
            print(f"Training Elastic Net model with features: {features}")
            predictor = ElasticNetPredictor(features=features)
        elif args.model == 'ensemble':
            print(f"Training Ensemble model (XGBoost + Elastic Net) with features: {features}")
            predictor = EnsemblePredictor(features=features)
        else:
            raise ValueError(f"Unknown model: {args.model}")

        predictor.train(train_df, tune=True)  # Enable tuning for manual training
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
        predict_upcoming_fixtures(use_full_schedule=args.all, model_type=args.model)
    elif args.backtest:
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
        backtester = Backtester(
            initial_bankroll=1000,
            staking_method='kelly',
            model_type=args.model,
            max_odds=args.max_odds
        )

        # Parse season argument
        if args.season.lower() == 'all':
            # Auto-detect available complete seasons from the data
            # Get unique years from the data
            df_temp = df.copy()
            df_temp['Date'] = pd.to_datetime(df_temp['Date'])

            # Drop rows with invalid dates
            df_temp = df_temp.dropna(subset=['Date'])

            df_temp['Year'] = df_temp['Date'].dt.year
            df_temp['Month'] = df_temp['Date'].dt.month

            # A season spans from August (month 8) of year N to July (month 7) of year N+1
            # Find seasons that have COMPLETED with actual results (not just scheduled fixtures)
            available_seasons = []
            current_date = pd.Timestamp.now()

            for year in sorted(df_temp['Year'].unique()):
                year = int(year)  # Convert to int to avoid float formatting issues
                season_end = pd.Timestamp(f"{year + 1}-07-31")

                # Only include seasons that have ended (completed)
                if season_end < current_date:
                    first_half = df_temp[(df_temp['Year'] == year) & (df_temp['Month'] >= 8)]
                    second_half = df_temp[(df_temp['Year'] == year + 1) & (df_temp['Month'] <= 7)]

                    # Verify we have substantial data for both halves
                    if len(first_half) > 50 and len(second_half) > 50:
                        season_str = f"{year}-{year + 1}"
                        available_seasons.append(season_str)

            seasons = available_seasons

            print(f"Auto-detected {len(seasons)} complete seasons: {', '.join(seasons)}")
            if not seasons:
                print("No complete seasons found in data!")
                return
            backtester.run_multi_season(df, features, seasons=seasons, retrain_every=args.retrain_every)
        elif ',' in args.season:
            # Multiple seasons specified
            seasons = [s.strip() for s in args.season.split(',')]
            print(f"Running backtest for {len(seasons)} seasons: {', '.join(seasons)}")
            backtester.run_multi_season(df, features, seasons=seasons, retrain_every=args.retrain_every)
        else:
            # Single season
            print(f"Running backtest for season {args.season}...")
            backtester.run(df, features, start_season=args.season, retrain_every=args.retrain_every)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
