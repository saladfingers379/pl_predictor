import argparse

def main():
    parser = argparse.ArgumentParser(description="Premier League Match Odds Predictor")
    parser.add_argument('--predict', action='store_true', help='Predict upcoming fixtures')
    parser.add_argument('--backtest', action='store_true', help='Run backtest on historical data')
    
    args = parser.parse_args()
    
    if args.predict:
        print("Running prediction...")
        # TODO: Import and run prediction logic
    elif args.backtest:
        print("Running backtest...")
        # TODO: Import and run backtest logic
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
