import pandas as pd
import numpy as np

def calculate_league_table(df):
    """
    Calculates the league table based on match results.
    """
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    table = pd.DataFrame(index=teams, columns=['P', 'W', 'D', 'L', 'GF', 'GA', 'GD', 'Pts'])
    table[:] = 0
    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
            continue
            
        hg = int(row['FTHG'])
        ag = int(row['FTAG'])
        
        table.loc[home, 'P'] += 1
        table.loc[away, 'P'] += 1
        table.loc[home, 'GF'] += hg
        table.loc[away, 'GF'] += ag
        table.loc[home, 'GA'] += ag
        table.loc[away, 'GA'] += hg
        
        if hg > ag:
            table.loc[home, 'W'] += 1
            table.loc[home, 'Pts'] += 3
            table.loc[away, 'L'] += 1
        elif ag > hg:
            table.loc[away, 'W'] += 1
            table.loc[away, 'Pts'] += 3
            table.loc[home, 'L'] += 1
        else:
            table.loc[home, 'D'] += 1
            table.loc[away, 'D'] += 1
            table.loc[home, 'Pts'] += 1
            table.loc[away, 'Pts'] += 1
            
    table['GD'] = table['GF'] - table['GA']
    table = table.sort_values(by=['Pts', 'GD', 'GF'], ascending=False)
    return table

def calculate_rolling_stats(df, window=5):
    """
    Calculates rolling statistics (form) for each team.
    Returns a DataFrame with rolling stats for each match-team combination.
    """
    # Create a long-form dataframe where each row is a team-match
    home_df = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HS', 'AS', 'HST', 'AST']].copy()
    home_df.rename(columns={'HomeTeam': 'Team', 'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst', 'HS': 'ShotsFor', 'AS': 'ShotsAgainst', 'HST': 'ShotsOnTargetFor', 'AST': 'ShotsOnTargetAgainst'}, inplace=True)
    home_df['IsHome'] = 1
    home_df['Points'] = home_df['FTR'].map({'H': 3, 'D': 1, 'A': 0})
    
    away_df = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AS', 'HS', 'AST', 'HST']].copy()
    away_df.rename(columns={'AwayTeam': 'Team', 'FTAG': 'GoalsFor', 'FTHG': 'GoalsAgainst', 'AS': 'ShotsFor', 'HS': 'ShotsAgainst', 'AST': 'ShotsOnTargetFor', 'HST': 'ShotsOnTargetAgainst'}, inplace=True)
    away_df['IsHome'] = 0
    away_df['Points'] = away_df['FTR'].map({'A': 3, 'D': 1, 'H': 0})
    
    team_stats = pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])
    
    # Calculate rolling stats
    cols_to_roll = ['GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'ShotsOnTargetFor', 'ShotsOnTargetAgainst', 'Points']
    
    for col in cols_to_roll:
        # Calculate rolling mean
        # shift(1) ensures we use past data for current row
        # min_periods=1 allows calculation even with 1 data point
        team_stats[f'Rolling_{col}'] = team_stats.groupby('Team')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        
        # Forward fill to propagate "last known form" into the future (where goals are NaN)
        team_stats[f'Rolling_{col}'] = team_stats.groupby('Team')[f'Rolling_{col}'].ffill()
        
    return team_stats

def prepare_training_data(df, window=5, is_training=True):
    """
    Prepares the final training dataset by merging rolling stats back to the match data.
    """
    team_stats = calculate_rolling_stats(df, window)
    
    # Merge Home Team Stats
    df_merged = pd.merge(df, team_stats[['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Points']], 
                         left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left', suffixes=('', '_Home'))
    
    df_merged.rename(columns={
        'Rolling_GoalsFor': 'Home_Rolling_GF',
        'Rolling_GoalsAgainst': 'Home_Rolling_GA',
        'Rolling_Points': 'Home_Rolling_Pts'
    }, inplace=True)
    df_merged.drop(columns=['Team'], inplace=True)
    
    # Merge Away Team Stats
    df_merged = pd.merge(df_merged, team_stats[['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Points']], 
                         left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left', suffixes=('', '_Away'))
    
    df_merged.rename(columns={
        'Rolling_GoalsFor': 'Away_Rolling_GF',
        'Rolling_GoalsAgainst': 'Away_Rolling_GA',
        'Rolling_Points': 'Away_Rolling_Pts'
    }, inplace=True)
    df_merged.drop(columns=['Team'], inplace=True)
    
    # Encode Target
    if 'FTR' in df_merged.columns:
        df_merged['Target'] = df_merged['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    else:
        df_merged['Target'] = np.nan
    
    # Drop rows with NaN (first few games)
    # If training, we need Target. If predicting, we don't.
    subset_cols = ['Home_Rolling_GF', 'Away_Rolling_GF']
    if is_training:
        subset_cols.append('Target')
        
    df_merged.dropna(subset=subset_cols, inplace=True)
    
    return df_merged

def calculate_elo(df):
    """
    Calculates ELO ratings for teams.
    """
    # Placeholder for ELO calculation
    return df
