import pandas as pd
import numpy as np
from src.models import XGBoostPredictor
from src.feature_engineering import prepare_training_data

class Backtester:
    def __init__(self, initial_bankroll=1000, staking_method='flat', kelly_fraction=0.25, flat_stake=10):
        """
        Initialize the Backtester.

        Args:
            initial_bankroll (float): Starting bankroll amount
            staking_method (str): 'flat' for fixed stakes or 'kelly' for Kelly criterion
            kelly_fraction (float): Fraction of Kelly to use (0.25 = quarter Kelly)
            flat_stake (float): Fixed stake amount when using flat staking
        """
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history = []
        self.staking_method = staking_method
        self.kelly_fraction = kelly_fraction
        self.flat_stake = flat_stake

    def run(self, df, features, start_season='2023-2024', retrain_every=50):
        """
        Runs a walk-forward backtest.
        Trains on all data prior to a match week, predicts that week, and tracks P&L.

        Args:
            df (pd.DataFrame): Historical match data
            features (list): List of feature names to use
            start_season (str): Season to start backtesting from (e.g., '2023-2024', '23/24', '2022-2023')
            retrain_every (int): Number of matches to wait before retraining the model.
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
        
        # Walk-forward loop
        predictor = XGBoostPredictor(features=features)
        
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
            
            # Predict for this chunk using current model
            probs = predictor.predict_proba(chunk)
            
            chunk_results = chunk.copy()
            chunk_results['Prob_Home'] = probs[:, 0]
            chunk_results['Prob_Draw'] = probs[:, 1]
            chunk_results['Prob_Away'] = probs[:, 2]
            all_predictions.append(chunk_results)
            
            # Retrain model with new data available up to the end of this chunk
            current_end_date = chunk['Date'].max()
            new_train_data = full_data[full_data['Date'] <= current_end_date]
            
            # Only retrain if there's more data and we aren't at the end
            if i + retrain_every < len(test_data):
                print(f"Retraining model with data up to {current_end_date}...")
                predictor.train(new_train_data)
        
        results = pd.concat(all_predictions)
        
        # Simulate Betting (Value Betting Strategy)
        self.simulate_betting(results)
        
    def simulate_betting(self, df):
        """
        Simulates betting based on calculated edge.
        """
        bets_placed = 0
        total_staked = 0
        total_return = 0
        wins = 0
        
        for index, row in df.iterrows():
            # Check if odds exist
            if pd.isna(row.get('B365H')) or pd.isna(row.get('B365D')) or pd.isna(row.get('B365A')):
                continue
                
            # Calculate edges
            edge_home = row['Prob_Home'] - (1 / row['B365H'])
            edge_draw = row['Prob_Draw'] - (1 / row['B365D'])
            edge_away = row['Prob_Away'] - (1 / row['B365A'])

            # Place bet on highest edge if positive
            max_edge = max(edge_home, edge_draw, edge_away)

            # Determine bet size based on staking method
            if self.staking_method == 'kelly':
                if max_edge == edge_home:
                    p = row['Prob_Home']
                    odds = row['B365H']
                elif max_edge == edge_draw:
                    p = row['Prob_Draw']
                    odds = row['B365D']
                else:
                    p = row['Prob_Away']
                    odds = row['B365A']

                b = odds - 1
                q = 1 - p
                kelly_fraction_optimal = (b * p - q) / b
                bet_size = max(0, self.kelly_fraction * kelly_fraction_optimal * self.bankroll)
                bet_size = min(bet_size, self.bankroll * 0.10)
            else:
                bet_size = self.flat_stake
            
            if max_edge > 0.05: # 5% edge threshold
                bets_placed += 1
                total_staked += bet_size
                self.bankroll -= bet_size
                
                won = False
                if max_edge == edge_home:
                    if row['FTR'] == 'H':
                        winnings = bet_size * row['B365H']
                        total_return += winnings
                        self.bankroll += winnings
                        won = True
                elif max_edge == edge_draw:
                    if row['FTR'] == 'D':
                        winnings = bet_size * row['B365D']
                        total_return += winnings
                        self.bankroll += winnings
                        won = True
                else:
                    if row['FTR'] == 'A':
                        winnings = bet_size * row['B365A']
                        total_return += winnings
                        self.bankroll += winnings
                        won = True
                        
                if won:
                    wins += 1

                # Track history only when a bet is placed
                self.history.append({'Date': row['Date'], 'Balance': self.bankroll})

        profit = total_return - total_staked
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        win_rate = (wins / bets_placed) * 100 if bets_placed > 0 else 0

        print("\n--- Backtest Results ---")
        print(f"Staking Method: {self.staking_method.upper()}" + (f" ({self.kelly_fraction:.0%} Kelly)" if self.staking_method == 'kelly' else f" (${self.flat_stake} per bet)"))
        print(f"Initial Bankroll: ${self.initial_bankroll:.2f}")
        print(f"Bets Placed: {bets_placed}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Staked: ${total_staked:.2f}")
        print(f"Total Return: ${total_return:.2f}")
        print(f"Profit: ${profit:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Bankroll: ${self.bankroll:.2f}")
        
        # Calculate Sharpe Ratio (daily returns)
        if self.history:
            history_df = pd.DataFrame(self.history)
            daily_returns = history_df.groupby('Date')['Balance'].last().pct_change().dropna()
            if not daily_returns.empty:
                sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) # Annualized
                print(f"Sharpe Ratio: {sharpe:.2f}")
            
            # Max Drawdown
            peak = history_df['Balance'].cummax()
            drawdown = (history_df['Balance'] - peak) / peak
            max_drawdown = drawdown.min() * 100
            print(f"Max Drawdown: {max_drawdown:.2f}%")
        
        # Plotting
        self.plot_balance()
        
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

