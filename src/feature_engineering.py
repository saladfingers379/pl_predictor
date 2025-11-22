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

def calculate_elo(df, k_factor=20, initial_elo=1500):
    """
    Calculates ELO ratings for teams over time.
    Returns a DataFrame with ELO ratings for Home and Away teams for each match.
    """
    # Sort by date to ensure correct order
    df = df.sort_values('Date').reset_index(drop=True)
    
    elo_dict = {}
    
    # Initialize ELOs
    teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel('K'))
    for team in teams:
        elo_dict[team] = initial_elo
        
    home_elos = []
    away_elos = []
    
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        
        # Get current ELOs
        home_elo = elo_dict.get(home, initial_elo)
        away_elo = elo_dict.get(away, initial_elo)
        
        home_elos.append(home_elo)
        away_elos.append(away_elo)
        
        # Skip if result is not available (future games)
        if pd.isna(row['FTHG']) or pd.isna(row['FTAG']):
            continue
            
        # Calculate expected score
        # Home advantage bonus (usually around 100 points)
        home_adv = 100
        dr = (home_elo + home_adv) - away_elo
        e_home = 1 / (1 + 10 ** (-dr / 400))
        e_away = 1 - e_home
        
        # Calculate actual score
        hg = row['FTHG']
        ag = row['FTAG']
        
        if hg > ag:
            s_home = 1
            s_away = 0
        elif ag > hg:
            s_home = 0
            s_away = 1
        else:
            s_home = 0.5
            s_away = 0.5
            
        # Update ELOs
        new_home_elo = home_elo + k_factor * (s_home - e_home)
        new_away_elo = away_elo + k_factor * (s_away - e_away)
        
        elo_dict[home] = new_home_elo
        elo_dict[away] = new_away_elo
        
    df['Home_ELO'] = home_elos
    df['Away_ELO'] = away_elos
    
    return df

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
    
    # Calculate Rest Days
    team_stats['Date'] = pd.to_datetime(team_stats['Date'])
    team_stats['LastMatchDate'] = team_stats.groupby('Team')['Date'].shift(1)
    team_stats['RestDays'] = (team_stats['Date'] - team_stats['LastMatchDate']).dt.days
    team_stats['RestDays'] = team_stats['RestDays'].fillna(7) # Default to 7 days if no previous match
    
    # Cap rest days at 14 to avoid skewing data with international breaks too much
    team_stats['RestDays'] = team_stats['RestDays'].clip(upper=14)

    # Calculate rolling stats (General Form)
    cols_to_roll = ['GoalsFor', 'GoalsAgainst', 'ShotsFor', 'ShotsAgainst', 'ShotsOnTargetFor', 'ShotsOnTargetAgainst', 'Points']
    
    for col in cols_to_roll:
        # General Form
        team_stats[f'Rolling_{col}'] = team_stats.groupby('Team')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        team_stats[f'Rolling_{col}'] = team_stats.groupby('Team')[f'Rolling_{col}'].ffill()
        
        # Home/Away Specific Form
        # We need to group by Team AND IsHome, but we want the rolling window to respect the sequence of ALL games?
        # No, usually "Home Form" means "Form in recent Home Games".
        
        # To do this efficiently, we can filter, calculate, and merge back, or use complex groupby
        # Let's do a separate calculation for Home/Away specific form
        
    return team_stats

def calculate_ha_specific_rolling_stats(df, window=5):
    """
    Calculates rolling stats specifically for Home games and Away games separately.
    """
    home_df = df[df['IsHome'] == 1].copy()
    away_df = df[df['IsHome'] == 0].copy()
    
    cols_to_roll = ['GoalsFor', 'GoalsAgainst', 'Points']
    
    for col in cols_to_roll:
        home_df[f'HomeRolling_{col}'] = home_df.groupby('Team')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        away_df[f'AwayRolling_{col}'] = away_df.groupby('Team')[col].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        
    return pd.concat([home_df, away_df]).sort_values(['Team', 'Date'])

def prepare_training_data(df, window=5, is_training=True):
    """
    Prepares the final training dataset by merging rolling stats back to the match data.
    """
    # 1. Calculate ELO
    df = calculate_elo(df)
    
    # 2. Calculate General Rolling Stats & Rest Days
    team_stats = calculate_rolling_stats(df, window)
    
    # 3. Calculate Home/Away Specific Rolling Stats
    # We need to re-merge or re-calculate. 
    # Let's refine the logic: calculate_rolling_stats returns a long DF.
    # We can split it, calc specific form, and concat.
    
    ha_stats = calculate_ha_specific_rolling_stats(team_stats, window)
    
    # Now merge back to the main match dataframe
    
    # Merge Home Team Stats
    cols_to_merge = ['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Points', 'RestDays', 'HomeRolling_GoalsFor', 'HomeRolling_GoalsAgainst', 'HomeRolling_Points']
    # Note: HomeRolling stats will be NaN for Away games in the long DF, and vice versa.
    # But for the Home Team in the main DF, we want their HomeRolling stats (since they are at home).
    
    # Actually, if a team is playing at Home, we want their recent Home Form.
    # If a team is playing Away, we want their recent Away Form.
    
    # Let's simplify:
    # ha_stats has 'HomeRolling_...' populated where IsHome=1
    # ha_stats has 'AwayRolling_...' populated where IsHome=0
    
    # We need to be careful. When predicting a Home game for Team A, we want Team A's recent Home stats.
    # So we merge on Team and Date.
    
    # Prepare Home Team Merge
    home_stats_subset = ha_stats[ha_stats['IsHome'] == 1][['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Points', 'RestDays', 'HomeRolling_GoalsFor', 'HomeRolling_GoalsAgainst', 'HomeRolling_Points']]
    
    df_merged = pd.merge(df, home_stats_subset, 
                         left_on=['Date', 'HomeTeam'], right_on=['Date', 'Team'], how='left')
    
    df_merged.rename(columns={
        'Rolling_GoalsFor': 'Home_Rolling_GF',
        'Rolling_GoalsAgainst': 'Home_Rolling_GA',
        'Rolling_Points': 'Home_Rolling_Pts',
        'RestDays': 'Home_RestDays',
        'HomeRolling_GoalsFor': 'Home_HomeForm_GF',
        'HomeRolling_GoalsAgainst': 'Home_HomeForm_GA',
        'HomeRolling_Points': 'Home_HomeForm_Pts'
    }, inplace=True)
    df_merged.drop(columns=['Team'], inplace=True)
    
    # Prepare Away Team Merge
    away_stats_subset = ha_stats[ha_stats['IsHome'] == 0][['Date', 'Team', 'Rolling_GoalsFor', 'Rolling_GoalsAgainst', 'Rolling_Points', 'RestDays', 'AwayRolling_GoalsFor', 'AwayRolling_GoalsAgainst', 'AwayRolling_Points']]
    
    df_merged = pd.merge(df_merged, away_stats_subset, 
                         left_on=['Date', 'AwayTeam'], right_on=['Date', 'Team'], how='left')
    
    df_merged.rename(columns={
        'Rolling_GoalsFor': 'Away_Rolling_GF',
        'Rolling_GoalsAgainst': 'Away_Rolling_GA',
        'Rolling_Points': 'Away_Rolling_Pts',
        'RestDays': 'Away_RestDays',
        'AwayRolling_GoalsFor': 'Away_AwayForm_GF',
        'AwayRolling_GoalsAgainst': 'Away_AwayForm_GA',
        'AwayRolling_Points': 'Away_AwayForm_Pts'
    }, inplace=True)
    df_merged.drop(columns=['Team'], inplace=True)
    
    # Encode Target
    if 'FTR' in df_merged.columns:
        df_merged['Target'] = df_merged['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    else:
        df_merged['Target'] = np.nan
    
    # Drop rows with NaN (first few games)
    # If training, we need Target. If predicting, we don't.
    subset_cols = ['Home_Rolling_GF', 'Away_Rolling_GF', 'Home_ELO', 'Away_ELO']
    if is_training:
        subset_cols.append('Target')
        
    df_merged.dropna(subset=subset_cols, inplace=True)
    
    # Fill specific form NaNs with general form if missing (e.g. first home game)
    df_merged['Home_HomeForm_GF'].fillna(df_merged['Home_Rolling_GF'], inplace=True)
    df_merged['Home_HomeForm_GA'].fillna(df_merged['Home_Rolling_GA'], inplace=True)
    df_merged['Home_HomeForm_Pts'].fillna(df_merged['Home_Rolling_Pts'], inplace=True)
    
    df_merged['Away_AwayForm_GF'].fillna(df_merged['Away_Rolling_GF'], inplace=True)
    df_merged['Away_AwayForm_GA'].fillna(df_merged['Away_Rolling_GA'], inplace=True)
    df_merged['Away_AwayForm_Pts'].fillna(df_merged['Away_Rolling_Pts'], inplace=True)
    
    return df_merged
