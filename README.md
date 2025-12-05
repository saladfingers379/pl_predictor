# Premier League Match Odds Predictor

This project predicts Premier League match outcomes and identifies value bets using a machine learning ensemble (XGBoost + ElasticNet). It utilizes historical data (2010-present) to generate "fair odds" and compares them against bookmaker odds to find profitable opportunities.

## Project Status
**Current Version:** Supports Ensemble Learning (Voting/Weighted), Rolling Window Backtesting, and Value Betting Simulation.

**Recent Findings:** 
- The model performs best as a "sniper" strategy, targeting heavy favorites (Odds ≤ 1.5).
- Backtesting across 13 seasons (2011-2025) shows positive ROI and stability with this strict threshold.
- An Ensemble of XGBoost (Gradient Boosting) and ElasticNet (Regularized Logistic Regression) provides the most robust predictions.

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   Create a `.env` file in the root directory if you intend to fetch live odds (optional for historical backtesting):
   ```
   ODDS_API_KEY=your_api_key_here
   ```

## Usage

The `main.py` script is the primary entry point.

### Common Commands

**1. Train the Model**
Train the selected model on all available historical data.
```bash
python main.py --train --model ensemble
```

**2. Predict Upcoming Fixtures**
Predict the probability and fair odds for upcoming games.
```bash
# Predict next immediate fixtures
python main.py --predict --model ensemble

# Predict ALL remaining fixtures in the schedule
python main.py --predict --all --model ensemble
```

**3. Run Backtests**
Simulate betting performance over historical seasons.
```bash
# Backtest a specific season
python main.py --backtest --season 2023-2024 --model ensemble

# Backtest ALL available complete seasons (2010-2024)
python main.py --backtest --season all --model ensemble --max-odds 1.5

# Backtest multiple specific seasons
python main.py --backtest --season 2022-2023,2023-2024 --model ensemble
```

**4. Update Data**
Fetch the latest live odds (The Odds API) and the full season schedule.
```bash
python main.py --update-data
```

**5. List Available Data**
See which seasons are available in your local dataset.
```bash
python main.py --list-seasons
```

### Command Line Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--train` | `False` | Train the model on historical data. |
| `--predict` | `False` | Predict upcoming fixtures using the trained model. |
| `--backtest` | `False` | Run a historical backtest to evaluate strategy. |
| `--all` | `False` | Used with `--predict`. If set, predicts all future scheduled games, not just the next batch. |
| `--season` | `2023-2024` | Season(s) to backtest. Accepts: <br> - Single: `2023-2024` <br> - Multiple: `2022-2023,2023-2024` <br> - All: `all` (Auto-detects complete seasons) |
| `--model` | `xgboost` | Model architecture to use: <br> - `xgboost`: Gradient Boosting Trees <br> - `elasticnet`: Regularized Logistic Regression <br> - `ensemble`: Weighted average of both (Recommended) |
| `--max-odds` | `2.0` | Maximum decimal odds to bet on. Limits strategy to favorites. Recommended: `1.5` for safety. |
| `--retrain-every` | `50` | **Backtesting only.** Number of matches to simulate before retraining the model with new data. |
| `--list-seasons` | `False` | Lists all complete seasons found in the data directory. |
| `--update-data` | `False` | Fetches latest odds and schedule from APIs. |

## Features
- **Ensemble Modeling**: Combines non-linear (XGBoost) and linear (ElasticNet) models to reduce variance.
- **Dynamic Feature Engineering**:
    - **ELO Ratings**: Updates after every match.
    - **Rolling Form**: Tracks goals and points over the last 5 games.
    - **Rest Days**: Accounts for team fatigue.
- **Smart Backtesting**:
    - **Walk-Forward Validation**: Prevents data leakage by training only on past data.
    - **Retraining Loops**: Simulates the model "learning" as the season progresses.
- **Betting Strategy**:
    - **Value Betting**: Only bets when Model Probability > Implied Probability + Edge.
    - **Kelly Criterion**: Optional staking method (defaults to flat staking currently).
    - **Odds Filters**: Configurable thresholds to avoid high-risk underdogs.

## Project Structure
- `main.py`: CLI entry point.
- `src/models.py`: XGBoost, ElasticNet, and Ensemble model definitions.
- `src/backtest.py`: Logic for running walk-forward simulations and calculating P&L.
- `src/feature_engineering.py`: Raw data transformation (ELO, Rolling stats).
- `src/data_loader.py`: Loading and cleaning of CSV data.
- `data/`: Stores historical CSVs and model outputs.