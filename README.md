# Premier League Match Odds Predictor

This project predicts Premier League match outcomes and "fair odds" using historical data and machine learning (XGBoost).

## Quick Start

### 1. Train the Model
Train the model on all available historical data. This uses the new features (ELO, Home/Away Form) and performs hyperparameter tuning.
```bash
python main.py --train
```

### 2. Run a Backtest
Simulate how the model would have performed in a past season.
```bash
# Backtest the 2023-2024 season
python main.py --backtest --season 2023-2024

# Backtest with more frequent retraining (every 10 games)
python main.py --backtest --season 2023-2024 --retrain-every 10
```

### 3. Predict Upcoming Fixtures
Predict the outcomes of future matches.
```bash
# Predict the next batch of fixtures
python main.py --predict

# Predict ALL future fixtures in the schedule
python main.py --predict --all
```

## Features
- **Dynamic ELO**: Teams have an ELO rating that updates after every match.
- **Home/Away Form**: Separate rolling statistics for home and away performance.
- **Rest Days**: Accounts for fatigue by tracking days since the last match.
- **Rolling Backtesting**: The model retrains periodically during backtesting to simulate real-world conditions.
- **Hyperparameter Tuning**: Automatically finds the best model parameters using `RandomizedSearchCV`.

## Project Structure

### `main.py`
The CLI entry point. Handles argument parsing and orchestrates the pipeline.

### `src/`
- **`feature_engineering.py`**: Calculates ELO, rolling stats, and rest days.
- **`models.py`**: Contains the `XGBoostPredictor` with tuning and training logic.
- **`backtest.py`**: The backtesting engine that simulates betting and calculates ROI, Sharpe Ratio, and Drawdown.
- **`fixture_fetcher.py`**: Fetches latest odds and schedule from APIs.
- **`data_loader.py`**: Loads and cleans historical CSV data.

## Setup
1.  Install dependencies: `pip install -r requirements.txt`
2.  Set API Key in `.env`: `ODDS_API_KEY=your_key`
