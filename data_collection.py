import os
import yfinance as yf
import pandas as pd
import time
from datetime import datetime

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Fetch S&P 500 tickers
try:
    SP500_TICKERS = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
except Exception as e:
    print(f"Error fetching S&P 500 tickers: {e}")
    SP500_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'AMD']
    print("Using fallback tickers:", SP500_TICKERS)

def get_sector(ticker):
    for _ in range(3):
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get('sector', 'Unknown')
        except Exception as e:
            print(f"Error fetching sector for {ticker}: {e}")
            time.sleep(1)
    return 'Unknown'

def download_data(ticker, start_date, end_date, max_retries=3):
    for attempt in range(max_retries):
        try:
            print(f"Downloading data for {ticker}...")
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date, auto_adjust=False)
            
            if data.empty:
                print(f"No data for {ticker}")
                return False
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Ensure all required columns
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = 0.0 if col != 'Date' else pd.NaT
            
            # Convert Date to date-only for merging
            data['Date'] = pd.to_datetime(data['Date']).dt.date
            
            # Initialize Dividends and Splits columns
            data['Dividends'] = 0.0
            data['Splits'] = 1.0
            
            # Add dividends if available
            dividends = stock.dividends
            if not dividends.empty:
                dividends = dividends.reset_index().rename(columns={'Date': 'Date', 'Dividends': 'Dividends'})
                dividends['Date'] = pd.to_datetime(dividends['Date']).dt.date
                data = data.merge(dividends[['Date', 'Dividends']], on='Date', how='left')
                data['Dividends'] = data['Dividends_y'].fillna(data['Dividends_x'])
                data = data.drop(columns=['Dividends_x', 'Dividends_y'])
            
            # Add splits if available
            splits = stock.splits
            if not splits.empty:
                splits = splits.reset_index().rename(columns={'Date': 'Date', 'Stock Splits': 'Splits'})
                splits['Date'] = pd.to_datetime(splits['Date']).dt.date
                data = data.merge(splits[['Date', 'Splits']], on='Date', how='left')
                data['Splits'] = data['Splits_y'].fillna(data['Splits_x'])
                data = data.drop(columns=['Splits_x', 'Splits_y'])
            
            # Add sector
            data['Sector'] = get_sector(ticker)
            
            # Select and order columns
            data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Splits', 'Sector']]
            
            # Convert Date back to datetime
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Save to CSV
            file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
            data.to_csv(file_path, index=False)
            print(f"Data saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error for {ticker} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"Failed to download data for {ticker}")
                return False
        time.sleep(1)

if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    for i, ticker in enumerate(SP500_TICKERS):
        ticker = ticker.replace('.', '-')  # Clean ticker symbol
        success = download_data(ticker, start_date, end_date)
        if not success:
            print(f"Skipping {ticker} due to persistent errors")
        if i % 20 == 0 and i > 0:
            print("Pausing to avoid rate limits...")
            time.sleep(30)