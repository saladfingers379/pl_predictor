import requests
import os
import pandas as pd
from datetime import datetime

# Manually read .env file
def load_env():
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value.strip('"\'')
    except FileNotFoundError:
        pass

load_env()
API_KEY = os.getenv('ODDS_API_KEY') or os.getenv('oddsapi')
BASE_URL = 'https://api.the-odds-api.com/v4/sports/soccer_epl/odds'

def get_upcoming_fixtures(api_key=None):
    """
    Fetches upcoming Premier League fixtures and odds from The Odds API.
    """
    if not api_key:
        api_key = API_KEY
        
    if not api_key:
        raise ValueError("API Key not found. Please set ODDS_API_KEY environment variable or pass it as an argument.")

    params = {
        'apiKey': api_key,
        'regions': 'uk',
        'markets': 'h2h',
        'bookmakers': 'betfair_ex_uk', # Betfair Exchange
        'oddsFormat': 'decimal'
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        return parse_odds_data(data)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def parse_odds_data(json_data):
    """
    Parses the JSON response from The Odds API into a DataFrame.
    """
    fixtures = []
    
    for event in json_data:
        home_team = event['home_team']
        away_team = event['away_team']
        commence_time = event['commence_time']
        
        # Find Betfair Exchange odds
        odds = {}
        for bookmaker in event.get('bookmakers', []):
            if bookmaker['key'] == 'betfair_ex_uk':
                for market in bookmaker.get('markets', []):
                    if market['key'] == 'h2h':
                        for outcome in market.get('outcomes', []):
                            if outcome['name'] == home_team:
                                odds['HomeOdds'] = outcome['price']
                            elif outcome['name'] == away_team:
                                odds['AwayOdds'] = outcome['price']
                            elif outcome['name'] == 'Draw':
                                odds['DrawOdds'] = outcome['price']
                break
        
        if odds:
            fixtures.append({
                'Date': datetime.fromisoformat(commence_time.replace('Z', '+00:00')),
                'HomeTeam': home_team,
                'AwayTeam': away_team,
                'HomeOdds': odds.get('HomeOdds'),
                'DrawOdds': odds.get('DrawOdds'),
                'AwayOdds': odds.get('AwayOdds')
            })
            
    return pd.DataFrame(fixtures)

def scrape_full_schedule():
    """
    Fetches the full Premier League schedule for the 2025/2026 season from fixturedownload.com.
    """
    url = "https://fixturedownload.com/feed/json/epl-2025"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        fixtures = []
        for match in data:
            # Format: {"MatchNumber":1,"RoundNumber":1,"DateUtc":"2025-08-16 12:30:00Z","Location":"...","HomeTeam":"...","AwayTeam":"...","Group":null,"HomeTeamScore":null,"AwayTeamScore":null}
            fixtures.append({
                'Date': datetime.fromisoformat(match['DateUtc'].replace('Z', '+00:00')),
                'HomeTeam': match['HomeTeam'],
                'AwayTeam': match['AwayTeam'],
                'Round': match['RoundNumber']
            })
            
        return pd.DataFrame(fixtures)
    except Exception as e:
        print(f"Error fetching full schedule: {e}")
        return pd.DataFrame(columns=['Date', 'HomeTeam', 'AwayTeam'])

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    try:
        print("Fetching upcoming fixtures from API...")
        df_api = get_upcoming_fixtures()
        print(df_api)
        if not df_api.empty:
            output_path = os.path.join('data', 'latest_odds.csv')
            df_api.to_csv(output_path, index=False)
            print(f"Saved odds to {output_path}")
        
        print("\nFetching full schedule...")
        df_schedule = scrape_full_schedule()
        print(df_schedule)
        if not df_schedule.empty:
            output_path = os.path.join('data', 'fixtures.csv')
            df_schedule.to_csv(output_path, index=False)
            print(f"Saved full schedule to {output_path}")
            
    except ValueError as e:
        print(e)
