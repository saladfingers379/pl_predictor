# Premier League Match Odds Predictor

This project predicts Premier League match outcomes and "fair odds" using historical data and machine learning (XGBoost).

## Project Structure & Usage

### 1. `main.py` - The Command Center
This is the entry point for the application. You run this file to execute different parts of the pipeline.

*   **Train Model**: `python main.py --train`
    *   Loads historical CSV data.
    *   Calculates features (form, goals, etc.).
    *   Trains the XGBoost model.
    *   Outputs validation accuracy and feature importance.
*   **Predict Fixtures**: `python main.py --predict`
    *   Loads the trained model and predicts upcoming games (next ~20 matches).
    *   Use `--all` to predict **ALL** future fixtures from the full schedule.
    *   Outputs "Fair Odds" and probabilities to `data/predictions.csv`.
*   **Backtest**: `python main.py --backtest`
    *   Simulates betting on past seasons (e.g., 2023/24).
    *   Calculates ROI, Profit, and Win Rate.
    *   **Visualizes** bankroll performance over time (graph).

### 2. `src/fixture_fetcher.py` - Data Sourcing
*   **Purpose**: Gets the schedule and current bookmaker odds.
*   **Run**: `python src/fixture_fetcher.py`
*   **Output**:
    *   `data/latest_odds.csv`: Upcoming matches with Betfair Exchange odds (from The Odds API).
    *   `data/fixtures.csv`: Full 2025/26 season schedule (scraped).

### 3. `src/data_loader.py` - Data Ingestion
*   **Purpose**: Reads the raw CSV files from `Historical_data_from_football_dot_co_dot_uk/`.
*   **Key Function**: `load_historical_data()`
    *   Combines all season files into one DataFrame.
    *   Cleans column names (removes whitespace).
    *   Standardizes team names.

### 4. `src/feature_engineering.py` - Feature Creation
*   **Purpose**: Transforms raw match results into predictive features.
*   **Key Function**: `prepare_training_data()`
    *   Calculates **Rolling Stats** (last 5 games): Goals For, Goals Against, Points.
    *   **Future Predictions**: For games far in the future, the model assumes "Current Form" persists (stats are forward-filled).
    *   Creates the final dataset used by XGBoost.
    *   Target: `Home Win (0)`, `Draw (1)`, `Away Win (2)`.

### 5. `src/models.py` - Machine Learning
*   **Purpose**: Defines the XGBoost model wrapper.
*   **Key Class**: `XGBoostPredictor`
    *   `train(df)`: Trains the model.
    *   `predict_proba(df)`: Returns probabilities (e.g., Home: 0.45, Draw: 0.25, Away: 0.30).
    *   `get_feature_importance()`: Visualizes which stats matter most (e.g., "Away Form" vs "Home Goals").

## Setup
1.  Install dependencies: `pip install -r requirements.txt`
2.  Set API Key in `.env`: `ODDS_API_KEY=your_key`
