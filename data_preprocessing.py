import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def add_calendar_features(df):
    df['DayOfWeek'] = df.index.dayofweek
    df['Month'] = df.index.month
    df['IsMonthEnd'] = df.index.is_month_end.astype(int)
    df['IsQuarterEnd'] = df.index.is_quarter_end.astype(int)
    return df

def add_lag_features(df, lag_days=5):
    df['Return'] = df['Close'].pct_change()
    for lag in range(1, lag_days + 1):
        df[f'Return_lag_{lag}'] = df['Return'].shift(lag)
    df.fillna(0, inplace=True)
    return df

def add_volatility(df, window=20):
    try:
        df['Volatility'] = df['Return'].rolling(window=window).std() * np.sqrt(252)
        if df['Volatility'].isna().all():
            df['Volatility'] = 0.0  # Default for invalid data
        else:
            df['Volatility'].fillna(df['Volatility'].mean(), inplace=True)
        return df
    except Exception as e:
        print(f"Error in add_volatility: {e}")
        df['Volatility'] = 0.0  # Fallback
        return df

def save_scaler(scaler, filename='scaler.save'):
    os.makedirs('scalers', exist_ok=True)
    joblib.dump(scaler, f"scalers/{filename}")

def extended_preprocess(df, ticker='unknown'):
    try:
        df.index = pd.to_datetime(df.index)
        df = df.asfreq('B')
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        # Ensure numeric columns
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Splits']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().all():
                raise ValueError(f"Column {col} contains all NaNs after conversion")

        df = df.astype({
            'Open': float, 'High': float, 'Low': float, 'Close': float,
            'Volume': int, 'Dividends': float, 'Splits': float
        })

        df = add_calendar_features(df)
        df = add_lag_features(df, lag_days=5)
        df = add_volatility(df)

        features_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility']
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

        save_scaler(scaler, f"scaler_{ticker}.save")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"Error in extended_preprocess for {ticker}: {e}")
        return None