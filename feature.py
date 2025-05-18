import os
import pandas as pd
import numpy as np
import logging

# Import your feature_engineering function
# Assuming add_feature.py is in the same directory

from add_feature import feature_engineering

# Setup
PREPROCESSED_DIR = "preprocessed_data"
OUTPUT_DIR = "featured_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename='add_features_all.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_preprocessed_data(ticker):
    """Load preprocessed CSV and validate structure."""
    try:
        file_path = os.path.join(PREPROCESSED_DIR, f"{ticker}.csv")
        df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
        expected_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Splits', 'Sector',
                        'DayOfWeek', 'Month', 'IsMonthEnd', 'IsQuarterEnd', 'Return',
                        'Return_lag_1', 'Return_lag_2', 'Return_lag_3', 'Return_lag_4', 'Return_lag_5', 'Volatility'}
        if not expected_cols.issubset(df.columns):
            logging.error(f"Missing columns in {ticker}.csv: {expected_cols - set(df.columns)}")
            return None
        logging.info(f"Loaded preprocessed data for {ticker}")
        return df
    except Exception as e:
        logging.error(f"Error loading {ticker}: {e}")
        return None

def add_features_ticker(ticker):
    """Apply feature engineering to a single ticker using your function."""
    # Load data
    df = load_preprocessed_data(ticker)
    if df is None:
        return None
    
    # Apply your feature_engineering function
    try:
        df = feature_engineering(df)
        logging.info(f"Added features for {ticker} with {len(df)} rows")
        return df
    except Exception as e:
        logging.error(f"Feature engineering failed for {ticker}: {e}")
        return None

def main():
    """Add features to all tickers and save results."""
    # Get list of tickers from CSV files
    tickers = [f.replace('.csv', '') for f in os.listdir(PREPROCESSED_DIR) if f.endswith('.csv')]
    if not tickers:
        logging.error("No CSV files found in preprocessed_data/")
        print("Error: No CSV files found in preprocessed_data/")
        return
    
    featured_data = []
    for i, ticker in enumerate(tickers):
        print(f"Adding features to {ticker} ({i+1}/{len(tickers)})...")
        df = add_features_ticker(ticker)
        if df is not None:
            # Save individual featured CSV
            output_path = os.path.join(OUTPUT_DIR, f"{ticker}.csv")
            df.to_csv(output_path)
            logging.info(f"Saved featured data for {ticker} to {output_path}")
            featured_data.append(df.assign(ticker=ticker))
    
    if not featured_data:
        logging.error("No data processed successfully")
        print("Error: No data processed successfully")
        return
    
    # Save consolidated dataset
    consolidated_df = pd.concat(featured_data, axis=0)
    output_path = os.path.join(OUTPUT_DIR, "consolidated_featured.csv")
    consolidated_df.to_csv(output_path)
    logging.info(f"Saved consolidated featured data to {output_path} with {len(consolidated_df)} rows")
    print(f"Saved consolidated featured data to {output_path}")
    
    # Log summary
    logging.info(f"Processed {len(featured_data)} tickers, total rows: {len(consolidated_df)}")
    print(f"Processed {len(featured_data)} tickers, total rows: {len(consolidated_df)}")

if __name__ == "__main__":
    main()