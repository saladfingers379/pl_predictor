import pandas as pd
import os
import glob

def load_historical_data(directory_path):
    """
    Loads all CSV files from the specified directory and concatenates them.
    Assumes files are in the format provided by football-data.co.uk.
    """
    all_files = glob.glob(os.path.join(directory_path, "*.csv"))
    df_list = []
    
    for filename in all_files:
        try:
            # football-data.co.uk files sometimes have encoding issues or trailing commas
            df = pd.read_csv(filename, encoding='ISO-8859-1', on_bad_lines='skip')
            
            # Clean column names (strip whitespace)
            df.columns = df.columns.str.strip()
            
            # Strip whitespace from all string columns
            df_obj = df.select_dtypes(['object'])
            df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
            
            # Add a season column if not present (can be inferred from filename or date)
            # For now, we'll just load the raw data
            df_list.append(df)
            print(f"Loaded {filename} with shape {df.shape}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            
    if not df_list:
        raise ValueError("No CSV files found in the specified directory.")
        
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Convert Date to datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True, errors='coerce')
    
    # Sort by date
    combined_df = combined_df.sort_values('Date')
    
    return combined_df

def standardize_teams(df):
    """
    Standardizes team names to ensure consistency across seasons and sources.
    """
    # TODO: Add specific mappings if discrepancies are found
    # Example: 'Man United' -> 'Man Utd'
    
    # Strip whitespace
    df['HomeTeam'] = df['HomeTeam'].str.strip()
    df['AwayTeam'] = df['AwayTeam'].str.strip()
    
    return df
