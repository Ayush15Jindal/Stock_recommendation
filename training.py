import pandas as pd
import glob
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load and consolidate CSVs
def load_ticker_data(data_path="sentiment_data/*.csv"):
    data_files = glob.glob(data_path)
    if not data_files:
        raise ValueError("No CSV files found in sentiment_data/")
    
    all_data = []
    for file in data_files:
        ticker = file.split("/")[-1].split("\\")[-1].replace(".csv", "").upper()
        if not (2 <= len(ticker) <= 5 and ticker.isalpha()):
            print(f"Skipping invalid ticker: {ticker}")
            continue
        try:
            df = pd.read_csv(file, skiprows=0, on_bad_lines="warn")
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
            invalid_dates = df["Date"].isna().sum()
            if invalid_dates > 0:
                print(f"{ticker}: Dropped {invalid_dates} rows with invalid dates")
                df = df.dropna(subset=["Date"])
            print(f"{ticker}: Rows = {len(df)}, Latest date = {df['Date'].max()}")
            df["Ticker"] = ticker
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded. Check CSV files.")
    
    data = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(data)} rows from {len(all_data)} tickers")
    print(f"Latest date after concatenation: {data['Date'].max()}")
    return data

# Preprocessing
def preprocess_data(data):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Ensure Date is datetime
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        print(f"Warning: Dropped {data['Date'].isna().sum()} rows with invalid dates")
        data = data.dropna(subset=["Date"])
    
    # Check date range
    latest_date = data["Date"].max()
    print(f"Latest date in data: {latest_date}")
    if latest_date < pd.to_datetime("2025-05-01"):
        print("Warning: Data is outdated. Expected dates up to ~2025-05-18.")
    
    # Handle missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    data["Sector"] = data["Sector"].fillna(data["Sector"].mode()[0])
    
    # Validate volatility
    if (data["Volatility"] < 0).any():
        print("Warning: Negative volatility detected. Setting to 0.")
        data.loc[data["Volatility"] < 0, "Volatility"] = 0
    
    # Encode Sector
    print("Unique sectors:", data["Sector"].unique())
    print("Number of sectors:", len(data["Sector"].unique()))
    data = pd.get_dummies(data, columns=["Sector"], drop_first=False)
    
    # Create risk category
    try:
        data["Risk_Category"] = pd.qcut(data["Volatility"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    except:
        print("Warning: Could not create Risk_Category. Using median Volatility as fallback")
        median_vol = data["Volatility"].median()
        data["Risk_Category"] = np.where(data["Volatility"] <= median_vol, "Low", "High")
    
    # Adjust Close for splits and dividends
    if "Adjusted_Close" in data.columns:
        data["Close_Adjusted"] = data["Adjusted_Close"]
    else:
        print("Warning: No Adjusted_Close. Approximating with Close, Splits, Dividends")
        data["Close_Adjusted"] = data["Close"]
        data["Split_Factor"] = 1 / (1 + data["Splits"].fillna(0))
        data["Split_Factor"] = data.groupby("Ticker")["Split_Factor"].cumprod()
        data["Close_Adjusted"] = data["Close_Adjusted"] * data["Split_Factor"]
        data["Close_Adjusted"] = data["Close_Adjusted"] + data["Dividends"].fillna(0).cumsum()
    
    # Create horizon-specific targets
    data = data.sort_values(["Ticker", "Date"])
    for horizon, days in [("1d", 1), ("1m", 30), ("6m", 180)]:
        data[f"target_{horizon}"] = data.groupby("Ticker")["Close_Adjusted"].shift(-days) / data["Close_Adjusted"] - 1
    
    # Clip extreme targets
    for horizon in ["1d", "1m", "6m"]:
        data[f"target_{horizon}"] = data[f"target_{horizon}"].clip(-0.5, 0.5)
    
    # Drop rows with NaN targets or key features
    initial_rows = len(data)
    key_features = ["Close_Adjusted", "Volatility", "RSI", "sentiment"]
    data = data.dropna(subset=[f"target_{horizon}" for horizon in ["1d", "1m", "6m"]] + key_features)
    dropped_rows = initial_rows - len(data)
    print(f"Dropped {dropped_rows} rows due to missing targets/features. Remaining: {len(data)}")
    if dropped_rows > 0:
        print("Missing value counts before dropping:")
        print(data[[f"target_{h}" for h in ["1d", "1m", "6m"]] + key_features].isna().sum())
    
    # Check target distribution
    for horizon in ["1d", "1m", "6m"]:
        print(f"Target {horizon} stats: mean={data[f'target_{horizon}'].mean():.4f}, "
              f"std={data[f'target_{horizon}'].std():.4f}, "
              f"min={data[f'target_{horizon}'].min():.4f}, "
              f"max={data[f'target_{horizon}'].max():.4f}")
    
    # Verify final date range
    print("Rows per ticker after preprocessing:")
    print(data.groupby("Ticker")["Date"].agg(["count", "min", "max"]))
    print("Latest date after preprocessing:", data["Date"].max())
    
    return data

# Train model
def train_model(data, horizon, features):
    data = data.sort_values("Date")
    train_end = pd.to_datetime("2023-12-31")
    test_end = pd.to_datetime("2024-12-31")
    train_data = data[data["Date"] <= train_end]
    test_data = data[(data["Date"] > train_end) & (data["Date"] <= test_end)]
    
    if len(train_data) < 1000 or len(test_data) < 100:
        raise ValueError(f"Insufficient data for {horizon}: train={len(train_data)}, test={len(test_data)}")
    
    X_train = train_data[features]
    y_train = train_data[f"target_{horizon}"]
    X_test = test_data[features]
    y_test = test_data[f"target_{horizon}"]
    
    print(f"Training {horizon}: {len(X_train)} train samples, {len(X_test)} test samples")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE for {horizon}: {mse:.4f}")
    print(f"Test RMSE: {np.sqrt(mse):.4f}")
    
    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print(f"Top 5 features for {horizon}:")
    print(importance.head())
    
    return model, scaler

# Main execution
if __name__ == "__main__":
    try:
        data = load_ticker_data()
        data = preprocess_data(data)
        data.to_csv("consolidated_sentiment.csv", index=False)
        
        features = [
            "Open", "High", "Low", "Close_Adjusted", "Volume", "Dividends", "Splits",
            "DayOfWeek", "Month", "IsMonthEnd", "IsQuarterEnd",
            "Return", "Return_lag_1", "Return_lag_2", "Return_lag_3", "Return_lag_4", "Return_lag_5",
            "Volatility", "SMA_14", "EMA_14", "RSI", "MACD", "MACD_signal", "MACD_hist", "sentiment"
        ] + [col for col in data.columns if col.startswith("Sector_")]
        
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        for horizon in ["1d", "1m", "6m"]:
            print(f"\nTraining model for {horizon} horizon...")
            model, scaler = train_model(data, horizon, features)
            pickle.dump(model, open(f"xgb_model_{horizon}.pkl", "wb"))
            pickle.dump(scaler, open(f"scaler_{horizon}.pkl", "wb"))
            print(f"Model saved as xgb_model_{horizon}.pkl, Scaler saved as scaler_{horizon}.pkl")
        
        print("\nTraining complete. Use inference script for recommendations.")
        
    except Exception as e:
        print(f"Error: {e}")