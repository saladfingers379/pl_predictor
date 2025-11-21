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
    """
    # This requires a more complex implementation to track stats per team over time
    # For now, we will return the dataframe as is, but this is where
    # we would add columns like 'HomeTeam_Last5_Points', etc.
    return df

def calculate_elo(df):
    """
    Calculates ELO ratings for teams.
    """
    # Placeholder for ELO calculation
    return df
