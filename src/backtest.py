import pandas as pd
import numpy as np
from src.models import XGBoostPredictor, ElasticNetPredictor, EnsemblePredictor
from src.feature_engineering import prepare_training_data

class Backtester:
    def __init__(self, initial_bankroll=1000, staking_method='flat', kelly_fraction=0.25, flat_stake=10, model_type='xgboost', max_odds=2.0):
        """
        Initialize the Backtester.

        Args:
            initial_bankroll (float): Starting bankroll amount
            staking_method (str): 'flat' for fixed stakes or 'kelly' for Kelly criterion
            kelly_fraction (float): Fraction of Kelly to use (0.25 = quarter Kelly)
            flat_stake (float): Fixed stake amount when using flat staking
            model_type (str): 'xgboost' or 'elasticnet' (default: xgboost)
            max_odds (float): Maximum odds to bet on (default 2.0 - favorites only)
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history = []
        self.staking_method = staking_method
        self.kelly_fraction = kelly_fraction
        self.flat_stake = flat_stake
        self.model_type = model_type
        self.max_odds = max_odds

    def run_multi_season(self, df, features, seasons, retrain_every=50):
        """
        Runs backtests across multiple seasons and aggregates results.

        Args:
            df (pd.DataFrame): Historical match data
            features (list): List of feature names to use
            seasons (list): List of season strings to backtest (e.g., ['2022-2023', '2023-2024'])
            retrain_every (int): Number of matches to wait before retraining the model.
        """
        print("=" * 70)
        print(f"MULTI-SEASON BACKTEST: {len(seasons)} seasons")
        print("=" * 70)

        all_season_results = []

        for season in seasons:
            print(f"\n{'='*70}")
            print(f"Season: {season}")
            print(f"{'='*70}")

            # Reset bankroll and history for each season
            self.bankroll = self.initial_bankroll
            self.history = []

            # Run single season backtest
            season_results = self.run(df, features, start_season=season, retrain_every=retrain_every, silent=False)

            if season_results:
                season_results['season'] = season
                all_season_results.append(season_results)

        # Aggregate results across all seasons
        if all_season_results:
            self._print_aggregate_results(all_season_results)

        return all_season_results

    def run(self, df, features, start_season='2023-2024', retrain_every=50, silent=False):
        """
        Runs a walk-forward backtest for a single season.
        Trains on all data prior to a match week, predicts that week, and tracks P&L.

        Args:
            df (pd.DataFrame): Historical match data
            features (list): List of feature names to use
            start_season (str): Season to start backtesting from (e.g., '2023-2024', '23/24', '2022-2023')
            retrain_every (int): Number of matches to wait before retraining the model.
            silent (bool): If True, suppresses some output (for multi-season runs)

        Returns:
            dict: Backtest results summary
        """
        # Ensure data is sorted by date
        df = df.sort_values('Date').reset_index(drop=True)

        # Prepare full dataset with features
        full_data = prepare_training_data(df)

        # Parse season string to get start year
        if '-' in start_season:
            start_year = int(start_season.split('-')[0])
        elif '/' in start_season:
            year_part = start_season.split('/')[0]
            start_year = int(year_part) if len(year_part) == 4 else 2000 + int(year_part)
        else:
            start_year = int(start_season)

        # Define season boundaries
        # Season usually starts in August and ends in May/June of next year
        season_start_date = pd.to_datetime(f'{start_year}-08-01').tz_localize(None)
        season_end_date = pd.to_datetime(f'{start_year + 1}-07-31').tz_localize(None)
        
        if full_data['Date'].dt.tz is not None:
             full_data['Date'] = full_data['Date'].dt.tz_convert(None)
             
        # Filter for the specific season ONLY
        test_data = full_data[(full_data['Date'] >= season_start_date) & (full_data['Date'] <= season_end_date)]
        
        if test_data.empty:
            print(f"No data found for backtesting period {start_season} ({season_start_date.date()} to {season_end_date.date()}).")
            return
            
        print(f"Backtesting on {len(test_data)} matches for season {start_season}...")
        print(f"Using model: {self.model_type}")

        # Walk-forward loop - select model based on type
        if self.model_type == 'xgboost':
            predictor = XGBoostPredictor(features=features)
        elif self.model_type == 'elasticnet':
            predictor = ElasticNetPredictor(features=features)
        elif self.model_type == 'ensemble':
            predictor = EnsemblePredictor(features=features)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Initial training: Train on EVERYTHING before this season
        train_data = full_data[full_data['Date'] < season_start_date]
        print(f"Initial training set size: {len(train_data)}")
        
        if train_data.empty:
            print("Warning: No training data available before this season. Model will be untrained initially.")
            # This might crash XGBoost if we try to fit empty data.
            # In this case, we might need to skip training or fail.
            return

        predictor.train(train_data)
        
        all_predictions = []
        
        # Process test data in chunks to simulate rolling updates
        for i in range(0, len(test_data), retrain_every):
            chunk = test_data.iloc[i:i+retrain_every]

            # Predict for this chunk using current model (model was trained BEFORE this chunk)
            probs = predictor.predict_proba(chunk)

            chunk_results = chunk.copy()
            chunk_results['Prob_Home'] = probs[:, 0]
            chunk_results['Prob_Draw'] = probs[:, 1]
            chunk_results['Prob_Away'] = probs[:, 2]
            all_predictions.append(chunk_results)

            # After predictions, retrain with data now available (including this chunk's results)
            # This simulates: predict -> matches happen -> results known -> retrain -> predict next batch
            # Only retrain if there's more data coming
            if i + retrain_every < len(test_data):
                current_end_date = chunk['Date'].max()
                new_train_data = full_data[full_data['Date'] <= current_end_date]
                print(f"Retraining model with data up to {current_end_date} ({len(new_train_data)} matches)...")
                predictor.train(new_train_data)
        
        results = pd.concat(all_predictions)

        # Simulate Betting (Value Betting Strategy)
        backtest_results = self.simulate_betting(results)

        return backtest_results
        
    def simulate_betting(self, df):
        """
        Simulates betting based on calculated edge with improved strategy:
        1. Removes bookmaker overround before calculating edge
        2. Excludes draw bets (hardest to predict)
        3. Uses higher edge threshold
        4. Only bets on favorites (odds <= max_odds)
        """
        bets_placed = 0
        total_staked = 0
        total_return = 0
        wins = 0
        draws_skipped = 0
        odds_filtered = 0  # Track bets skipped due to odds being too high

        for _, row in df.iterrows():
            # Check if odds exist
            if pd.isna(row.get('B365H')) or pd.isna(row.get('B365D')) or pd.isna(row.get('B365A')):
                continue

            # Step 1: Calculate implied probabilities from bookmaker odds
            implied_home = 1 / row['B365H']
            implied_draw = 1 / row['B365D']
            implied_away = 1 / row['B365A']

            # Step 2: Remove bookmaker overround (margin)
            # The sum of implied probabilities > 1.0 is the bookmaker's margin
            total_implied = implied_home + implied_draw + implied_away

            # Normalize to get fair implied probabilities
            fair_implied_home = implied_home / total_implied
            fair_implied_draw = implied_draw / total_implied
            fair_implied_away = implied_away / total_implied

            # Step 3: Calculate edges using fair odds
            edge_home = row['Prob_Home'] - fair_implied_home
            edge_draw = row['Prob_Draw'] - fair_implied_draw
            edge_away = row['Prob_Away'] - fair_implied_away

            # Step 4: Only consider Home/Away bets (exclude draws - model is bad at them)
            edges = {'home': edge_home, 'away': edge_away}
            max_edge_type = max(edges, key=edges.get)
            max_edge = edges[max_edge_type]

            # Track if we would have bet on a draw
            if edge_draw > max_edge and edge_draw > 0.10:
                draws_skipped += 1

            # Skip bet if bankroll is depleted
            if self.bankroll <= 0:
                continue

            # Step 5: Apply stricter edge threshold (10% instead of 5%)
            # This accounts for model uncertainty and ensures we only bet when confident
            if max_edge > 0.10:
                # Determine which outcome to bet on
                if max_edge_type == 'home':
                    p = row['Prob_Home']
                    odds = row['B365H']
                    outcome_to_bet = 'H'
                else:  # away
                    p = row['Prob_Away']
                    odds = row['B365A']
                    outcome_to_bet = 'A'

                # Step 6: Filter by maximum odds (only bet on favorites)
                if odds > self.max_odds:
                    odds_filtered += 1
                    continue

                # Calculate bet size based on staking method
                if self.staking_method == 'kelly':
                    b = odds - 1
                    q = 1 - p
                    kelly_fraction_optimal = (b * p - q) / b
                    bet_size = max(0, self.kelly_fraction * kelly_fraction_optimal * self.bankroll)
                    # Cap at 10% of bankroll for safety
                    bet_size = min(bet_size, self.bankroll * 0.10)
                else:
                    # For flat staking, ensure we don't bet more than available bankroll
                    bet_size = min(self.flat_stake, self.bankroll)

                # Skip if bet size is too small
                if bet_size <= 0:
                    continue

                # Place the bet
                bets_placed += 1
                total_staked += bet_size
                self.bankroll -= bet_size

                # Check if bet won
                won = row['FTR'] == outcome_to_bet
                if won:
                    winnings = bet_size * odds
                    total_return += winnings
                    self.bankroll += winnings
                    wins += 1

                # Track history
                self.history.append({'Date': row['Date'], 'Balance': self.bankroll})

        profit = total_return - total_staked
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        win_rate = (wins / bets_placed) * 100 if bets_placed > 0 else 0

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        print(f"Strategy: Favorites only (odds ≤ {self.max_odds:.2f}, no draws)")
        print(f"Edge Threshold: 10% (after removing overround)")
        print(f"Staking Method: {self.staking_method.upper()}" + (f" ({self.kelly_fraction:.0%} Kelly)" if self.staking_method == 'kelly' else f" (${self.flat_stake} per bet)"))
        print(f"\nInitial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"Final Bankroll: ${self.bankroll:.2f}")
        print(f"Profit/Loss: ${profit:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"\nBets Placed: {bets_placed}")
        print(f"  Draw Bets Skipped: {draws_skipped}")
        print(f"  Odds Too High (>{self.max_odds:.2f}): {odds_filtered}")
        print(f"Wins: {wins}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Staked: ${total_staked:.2f}")
        print(f"Total Return: ${total_return:.2f}")
        
        # Calculate Sharpe Ratio (daily returns)
        sharpe = 0.0
        max_drawdown = 0.0

        if self.history:
            history_df = pd.DataFrame(self.history)
            daily_returns = history_df.groupby('Date')['Balance'].last().pct_change().dropna()
            if not daily_returns.empty:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annualized
                print(f"\nSharpe Ratio: {sharpe:.2f}")

            # Max Drawdown
            peak = history_df['Balance'].cummax()
            drawdown = (history_df['Balance'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            print(f"Max Drawdown: {max_drawdown:.2f}%")
        else:
            print("\nNo bets placed - no performance metrics to calculate")

        print("=" * 60)

        # Plotting
        self.plot_balance()

        # Return results dictionary
        return {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': self.bankroll,
            'profit': profit,
            'roi': roi,
            'bets_placed': bets_placed,
            'wins': wins,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'draws_skipped': draws_skipped,
            'odds_filtered': odds_filtered
        }
        
    def _print_aggregate_results(self, all_season_results):
        """
        Prints aggregated results across multiple seasons.
        """
        print("\n" + "=" * 70)
        print("AGGREGATE RESULTS ACROSS ALL SEASONS")
        print("=" * 70)

        total_bets = sum(r['bets_placed'] for r in all_season_results)
        total_wins = sum(r['wins'] for r in all_season_results)
        total_staked = sum(r['total_staked'] for r in all_season_results)
        total_return = sum(r['total_return'] for r in all_season_results)
        total_profit = sum(r['profit'] for r in all_season_results)

        avg_roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        avg_win_rate = (total_wins / total_bets * 100) if total_bets > 0 else 0
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_season_results])
        avg_max_dd = np.mean([r['max_drawdown'] for r in all_season_results])

        print(f"\nSeasons Analyzed: {len(all_season_results)}")
        print(f"Total Bets Placed: {total_bets}")
        print(f"Total Wins: {total_wins}")
        print(f"Overall Win Rate: {avg_win_rate:.2f}%")
        print(f"\nTotal Staked: ${total_staked:.2f}")
        print(f"Total Return: ${total_return:.2f}")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Overall ROI: {avg_roi:.2f}%")
        print(f"\nAverage Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Max Drawdown: {avg_max_dd:.2f}%")

        print("\n" + "-" * 70)
        print("PER-SEASON BREAKDOWN:")
        print("-" * 70)
        print(f"{'Season':<15} {'Bets':<8} {'Win Rate':<12} {'ROI':<10} {'Profit':<12}")
        print("-" * 70)

        for r in all_season_results:
            print(f"{r['season']:<15} {r['bets_placed']:<8} {r['win_rate']:>8.2f}%  {r['roi']:>8.2f}%  ${r['profit']:>10.2f}")

        print("=" * 70)

    def plot_balance(self):
        """
        Plots the bankroll balance over time.
        """
        if not self.history:
            print("No betting history to plot.")
            return

        history_df = pd.DataFrame(self.history)
        # Group by date to get end-of-day balance
        daily_balance = history_df.groupby('Date')['Balance'].last().reset_index()

        # Add initial bankroll at the start of the backtest period
        first_date = daily_balance['Date'].min()
        initial_row = pd.DataFrame({'Date': [first_date], 'Balance': [self.initial_bankroll]})
        daily_balance = pd.concat([initial_row, daily_balance], ignore_index=True)
        daily_balance = daily_balance.sort_values('Date').reset_index(drop=True)

        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(daily_balance['Date'], daily_balance['Balance'], marker='o', markersize=3, linewidth=1.5)
        plt.title('Bankroll Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance ($)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', alpha=0.7, label='Initial Bankroll')

        # Set x-axis limits to only show the actual backtest period
        plt.xlim(daily_balance['Date'].min(), daily_balance['Date'].max())

        plt.legend()
        plt.tight_layout()
        
        # Save plot
        output_path = 'data/backtest_results.png'
        plt.savefig(output_path)
        print(f"Balance plot saved to {output_path}")
        # plt.show() # Optional: keep if running locally, but saving is safer for headless

