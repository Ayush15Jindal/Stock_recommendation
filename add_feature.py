import pandas as pd
import numpy as np
from data_preprocessing import extended_preprocess
import matplotlib.pyplot as plt

def add_SMA(df, window=14):
    df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
    df[f'SMA_{window}'].fillna(method='bfill', inplace=True)
    return df

def add_EMA(df, window=14):
    df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
    return df

def add_RSI(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    RS = gain / loss
    df['RSI'] = 100 - (100 / (1 + RS))
    df['RSI'].fillna(50, inplace=True)  # neutral RSI for NaNs
    return df

def add_MACD(df, span_short=12, span_long=26, span_signal=9):
    ema_short = df['Close'].ewm(span=span_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=span_long, adjust=False).mean()
    df['MACD'] = ema_short - ema_long
    df['MACD_signal'] = df['MACD'].ewm(span=span_signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df

def feature_engineering(df):
    df = add_SMA(df)
    df = add_EMA(df)
    df = add_RSI(df)
    df = add_MACD(df)
    return df


def plot_technical_indicators(df):
    plt.figure(figsize=(14, 10))

    # Price + SMA and EMA
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.plot(df.index, df['SMA_14'], label='SMA 14')
    plt.plot(df.index, df['EMA_14'], label='EMA 14')
    plt.title('Price with SMA and EMA')
    plt.legend()

    # RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI'], label='RSI')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.title('Relative Strength Index')
    plt.legend()

    # MACD
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['MACD'], label='MACD')
    plt.plot(df.index, df['MACD_signal'], label='Signal Line')
    plt.bar(df.index, df['MACD_hist'], label='MACD Histogram', color='grey')
    plt.title('MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Usage


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv('data/AAPL.csv', skiprows=4, header=None)

# Now manually set columns names, since header row was skipped
    df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'Dividends', 'Splits']

# Parse dates & set index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Preprocess data
    df = extended_preprocess(df)
    df = feature_engineering(df)
    plot_technical_indicators(df)