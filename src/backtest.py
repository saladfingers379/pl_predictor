import pandas as pd
import numpy as np
from src.models import XGBoostPredictor
from src.feature_engineering import prepare_training_data

class Backtester:
    def __init__(self, initial_bankroll=1000):
        self.initial_bankroll = initial_bankroll
        self.bankroll = initial_bankroll
        self.history = []
        
    def run(self, df, features, start_season='2022-2023'):
        """
        Runs a walk-forward backtest.
        Trains on all data prior to a match week, predicts that week, and tracks P&L.
        """
        # Ensure data is sorted by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Prepare full dataset with features
        full_data = prepare_training_data(df)
        
        # Filter for seasons to backtest (simple date filter for now)
        # Assuming 'Date' is datetime
        test_start_date = pd.to_datetime('2023-08-01').tz_localize(None) # Start of 23/24 season approx
        
        # If timezone aware, handle it. Let's assume UTC for now or convert.
        if full_data['Date'].dt.tz is not None:
             full_data['Date'] = full_data['Date'].dt.tz_convert(None)
             
        test_data = full_data[full_data['Date'] >= test_start_date]
        
        if test_data.empty:
            print("No data found for backtesting period.")
            return
            
        print(f"Backtesting on {len(test_data)} matches starting from {test_start_date}...")
        
        # Walk-forward loop
        # We'll retrain every week or just once? 
        # For true walk-forward, we should retrain periodically. 
        # Let's retrain every 50 matches to save time for this MVP, or just train on everything prior.
        
        # Better approach for MVP: Train on everything BEFORE the test season, then predict the test season.
        # Then we can implement rolling updates.
        
        train_data = full_data[full_data['Date'] < test_start_date]
        
        print(f"Initial training set size: {len(train_data)}")
        
        predictor = XGBoostPredictor(features=features)
        predictor.train(train_data)
        
        # Predict on test set
        probs = predictor.predict_proba(test_data)
        
        # Combine predictions with actuals and odds
        results = test_data.copy()
        results['Prob_Home'] = probs[:, 0]
        results['Prob_Draw'] = probs[:, 1]
        results['Prob_Away'] = probs[:, 2]
        
        # Simulate Betting (Value Betting Strategy)
        self.simulate_betting(results)
        
    def simulate_betting(self, df):
        """
        Simulates betting based on calculated edge.
        """
        bets_placed = 0
        total_staked = 0
        total_return = 0
        
        for index, row in df.iterrows():
            # Simple strategy: Bet if Model Probability > Implied Probability (1/Odds)
            # We need odds columns. Assuming B365H, B365D, B365A exist.
            
            # Check if odds exist
            if pd.isna(row.get('B365H')) or pd.isna(row.get('B365D')) or pd.isna(row.get('B365A')):
                continue
                
            # Calculate edges
            edge_home = row['Prob_Home'] - (1 / row['B365H'])
            edge_draw = row['Prob_Draw'] - (1 / row['B365D'])
            edge_away = row['Prob_Away'] - (1 / row['B365A'])
            
            bet_size = 10 # Flat stake
            
            # Place bet on highest edge if positive
            max_edge = max(edge_home, edge_draw, edge_away)
            
            if max_edge > 0.05: # 5% edge threshold
                bets_placed += 1
                total_staked += bet_size
                self.bankroll -= bet_size
                
                if max_edge == edge_home:
                    if row['FTR'] == 'H':
                        winnings = bet_size * row['B365H']
                        total_return += winnings
                        self.bankroll += winnings
                elif max_edge == edge_draw:
                    if row['FTR'] == 'D':
                        winnings = bet_size * row['B365D']
                        total_return += winnings
                        self.bankroll += winnings
                else:
                    if row['FTR'] == 'A':
                        winnings = bet_size * row['B365A']
                        total_return += winnings
                        self.bankroll += winnings
            
            # Track history (Date, Balance)
            # Only track if date changes or it's the last row to avoid too many points, 
            # but for simplicity let's track every bet or just group by date later.
            # Actually, let's just append current bankroll and date.
            self.history.append({'Date': row['Date'], 'Balance': self.bankroll})
                        
        profit = total_return - total_staked
        roi = (profit / total_staked) * 100 if total_staked > 0 else 0
        
        print("\n--- Backtest Results ---")
        print(f"Bets Placed: {bets_placed}")
        print(f"Total Staked: {total_staked}")
        print(f"Total Return: {total_return:.2f}")
        print(f"Profit: {profit:.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Final Bankroll: {self.bankroll:.2f}")
        
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
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(daily_balance['Date'], daily_balance['Balance'])
        plt.title('Bankroll Over Time')
        plt.xlabel('Date')
        plt.ylabel('Balance')
        plt.grid(True)
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', label='Initial Bankroll')
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("Balance plot displayed.")
