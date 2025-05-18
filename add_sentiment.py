import os
import pandas as pd
import time
import logging
from sentiment import fetch_news, analyze_sentiment, aggregate_daily_sentiment

# Setup
FEATURED_DIR = "featured_data"
OUTPUT_DIR = "sentiment_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='add_sentiment_all.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_featured_data(ticker):
    """Load featured CSV and validate structure."""
    try:
        file_path = os.path.join(FEATURED_DIR, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        expected_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Splits', 'Sector',
                        'DayOfWeek', 'Month', 'IsMonthEnd', 'IsQuarterEnd', 'Return',
                        'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_4', 'Return_lag_5',
                        'Volatility', 'SMA_14', 'EMA_14', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist'}
        if not expected_cols.issubset(df.columns):
            logging.error(f"Missing columns in {ticker}.csv: {expected_cols - set(df.columns)}")
            return None
        logging.info(f"Loaded featured data for {ticker}")
        return df
    except Exception as e:
        logging.error(f"Error loading {ticker}: {e}")
        return None

def add_sentiment_ticker(ticker):
    """Add sentiment scores to a single ticker using your functions."""
    # Load data
    df = load_featured_data(ticker)
    if df is None:
        return None
    
    # Get sentiment data
    try:
        sentiment_df = fetch_news(ticker, days=30)
        if not sentiment_df.empty:
            sentiment_df = analyze_sentiment(sentiment_df)
            sentiment_score = aggregate_daily_sentiment(sentiment_df)
            logging.info(f"30-day sentiment score for {ticker}: {sentiment_score}")
        else:
            sentiment_score = 0.0
            logging.warning(f"No sentiment data for {ticker}, setting sentiment to 0.0")
        
        # Assign sentiment to all rows
        df['sentiment'] = sentiment_score
        
        # Add target (next day's return)
        if df['Close'].std() < 1e-6:
            logging.error(f"Close values for {ticker} are constant or invalid")
            return None
        df['target'] = df['Close'].pct_change().shift(-1)
        
        # Clip extreme target values
        df['target'] = df['target'].clip(lower=-0.2, upper=0.2)  # Cap at ±20%
        
        # Drop rows with NaN target
        df = df.dropna(subset=['target'])
        
        # Log stats
        logging.info(f"Added sentiment for {ticker} with {len(df)} rows, sentiment: {df['sentiment'].iloc[0]}, target mean: {df['target'].mean()}, target std: {df['target'].std()}")
        return df
    except Exception as e:
        logging.error(f"Sentiment processing failed for {ticker}: {e}")
        return None

def main():
    """Add sentiment to all tickers and save results."""
    # Get list of tickers from CSV files
    tickers = [f.replace('.csv', '') for f in os.listdir(FEATURED_DIR) if f.endswith('.csv')]
    if not tickers:
        logging.error("No CSV files found in featured_data/")
        print("Error: No CSV files found in featured_data/")
        return
    
    sentiment_data = []
    for i, ticker in enumerate(tickers):
        print(f"Processing sentiment for {ticker} ({i+1}/{len(tickers)})...")
        df = add_sentiment_ticker(ticker)
        if df is not None:
            # Save individual sentiment CSV
            output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
            df.to_csv(output_path)
            logging.info(f"Saved sentiment data for {ticker} to {output_path}")
            sentiment_data.append(df.assign(ticker=ticker))
        
        # Pause to respect Alpha Vantage rate limits (5 calls/min)
        if (i + 1) % 5 == 0:
            print("Pausing to avoid rate limits...")
            time.sleep(60)
    
    if not sentiment_data:
        logging.error("No data processed successfully")
        print("Error: No data processed successfully")
        return
    
    # Save consolidated dataset
    consolidated_df = pd.concat(sentiment_data, axis=0)
    output_path = os.path.join(OUTPUT_DIR, "consolidated_sentiment.csv")
    consolidated_df.to_csv(output_path)
    logging.info(f"Saved consolidated sentiment data to {output_path} with {len(consolidated_df)} rows")
    print(f"Saved consolidated sentiment data to {output_path}")
    
    # Log summary
    logging.info(f"Processed {len(sentiment_data)} tickers, total rows: {len(consolidated_df)}")
    print(f"Processed {len(sentiment_data)} tickers, total rows: {len(consolidated_df)}")

if __name__ == "__main__":
    main()