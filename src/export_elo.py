import pandas as pd
from src.data_loader import load_historical_data
from src.feature_engineering import prepare_training_data

def export_current_elo():
    print("Loading historical data...")
    df = load_historical_data('Historical_data_from_football_dot_co_dot_uk')
    
    print("Calculating ELO...")
    # This function calculates ELO as part of feature engineering
    train_df = prepare_training_data(df)
    
    print("Exporting current ELO ratings...")
    # Get the last row for each team to get their latest ELO
    # We need to look at both Home and Away columns
    
    # Create a long format of team, date, elo
    home_elo = train_df[['Date', 'HomeTeam', 'Home_ELO']].rename(columns={'HomeTeam': 'Team', 'Home_ELO': 'ELO'})
    away_elo = train_df[['Date', 'AwayTeam', 'Away_ELO']].rename(columns={'AwayTeam': 'Team', 'Away_ELO': 'ELO'})
    
    all_elo = pd.concat([home_elo, away_elo])
    
    # Sort by date and take the last entry for each team
    latest_elo = all_elo.sort_values('Date').groupby('Team').last().reset_index()
    latest_elo = latest_elo[['Team', 'ELO']].sort_values('ELO', ascending=False)
    
    output_path = 'data/current_elo.csv'
    latest_elo.to_csv(output_path, index=False)
    print(f"Current ELO ratings saved to {output_path}")
    print(latest_elo.head(20))

if __name__ == "__main__":
    export_current_elo()
