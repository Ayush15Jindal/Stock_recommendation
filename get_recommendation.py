import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

# Load consolidated data
def load_data(file_path="consolidated_sentiment.csv"):
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")
    data = pd.read_csv(file_path)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    if data["Date"].isna().any():
        print(f"Warning: Dropped {data['Date'].isna().sum()} rows with invalid dates")
        data = data.dropna(subset=["Date"])
    return data

# Get available sectors from data
def get_available_sectors(data):
    sector_cols = [col for col in data.columns if col.startswith("Sector_")]
    return [col.replace("Sector_", "") for col in sector_cols]

# Validate and get user inputs
def get_user_inputs(data):
    sectors = get_available_sectors(data)
    horizons = ["1d", "1m", "6m"]
    risks = ["Low", "Medium", "High"]
    
    # Sector input
    print("\nAvailable sectors:", ", ".join(sectors + ["All"]))
    while True:
        sector = input("Enter sector (or 'All' for no filter): ").strip()
        if sector.lower() == "all":
            sector = None
            break
        if sector in sectors:
            break
        print(f"Invalid sector. Choose from {', '.join(sectors + ['All'])}")
    
    # Horizon input
    print("Available horizons: 1d (1 day), 1m (1 month), 6m (6 months)")
    while True:
        horizon = input("Enter horizon (1d, 1m, 6m): ").strip().lower()
        if horizon in horizons:
            break
        print(f"Invalid horizon. Choose from {', '.join(horizons)}")
    
    # Risk input
    print("Available risk levels: Low, Medium, High")
    while True:
        risk = input("Enter risk level (Low, Medium, High): ").strip().capitalize()
        if risk in risks:
            break
        print(f"Invalid risk. Choose from {', '.join(risks)}")
    
    return sector, horizon, risk

# Recommendation function
def get_recommendations(data, features, sector=None, horizon="1d", risk="Low", top_n=5):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")
    
    # Map horizon to model and scaler paths
    paths = {
        "1d": {"model": "xgb_model_1d.pkl", "scaler": "scaler_1d.pkl"},
        "1m": {"model": "xgb_model_1m.pkl", "scaler": "scaler_1m.pkl"},
        "6m": {"model": "xgb_model_6m.pkl", "scaler": "scaler_6m.pkl"}
    }
    
    if horizon not in paths:
        raise ValueError(f"Invalid horizon: {horizon}. Choose from {list(paths.keys())}")
    
    # Load model and scaler
    try:
        model = pickle.load(open(paths[horizon]["model"], "rb"))
        scaler = pickle.load(open(paths[horizon]["scaler"], "rb"))
    except Exception as e:
        raise ValueError(f"Error loading model/scaler for {horizon}: {e}")
    
    # Validate ticker names (e.g., exclude single-letter tickers)
    invalid_tickers = data[data["Ticker"].str.len() < 2]["Ticker"].unique()
    if invalid_tickers.size > 0:
        print(f"Warning: Invalid tickers found: {', '.join(invalid_tickers)}. Filtering out.")
        data = data[~data["Ticker"].isin(invalid_tickers)]
    
    # Validate volatility (must be non-negative)
    if (data["Volatility"] < 0).any():
        print("Warning: Negative volatility detected. Setting negative values to 0.")
        data.loc[data["Volatility"] < 0, "Volatility"] = 0
    
    # Filter by sector
    if sector:
        sector_col = [col for col in data.columns if col.startswith("Sector_") and sector.lower() in col.lower()]
        if sector_col:
            data = data[data[sector_col[0]] == 1]
        else:
            print(f"Warning: No data for sector {sector}. Returning empty recommendations.")
            return pd.DataFrame()
    
    # Filter by risk
    if risk:
        data = data[data["Risk_Category"] == risk]
    
    # Check if data remains after filtering
    if data.empty:
        print(f"No data available after filtering for sector={sector or 'All'}, risk={risk}.")
        return pd.DataFrame()
    
    # Get latest date
    latest_date = data["Date"].max()
    print(f"Using latest date: {latest_date.strftime('%Y-%m-%d')}")
    data = data[data["Date"] == latest_date]
    
    # Check if data remains after date filter
    if data.empty:
        print(f"No data available for latest date {latest_date.strftime('%Y-%m-%d')}.")
        return pd.DataFrame()
    
    # Prepare data
    X = data[features]
    X_scaled = scaler.transform(X)  # Apply scaling
    predictions = model.predict(X_scaled)
    
    # Create recommendation dataframe
    recommendations = pd.DataFrame({
        "Ticker": data["Ticker"],
        "Date": data["Date"],
        "Predicted_Return": predictions,
        "Volatility": data["Volatility"],
        "Risk_Category": data["Risk_Category"]
    })
    
    # Sort by predicted return
    recommendations = recommendations.sort_values("Predicted_Return", ascending=False)
    
    return recommendations.head(top_n)

# Main execution
if __name__ == "__main__":
    try:
        # Load data
        data = load_data()
        
        # Define features (must match training script)
        features = [
            "Open", "High", "Low", "Close_Adjusted", "Volume", "Dividends", "Splits",
            "DayOfWeek", "Month", "IsMonthEnd", "IsQuarterEnd",
            "Return", "Return_lag_1", "Return_lag_2", "Return_lag_3", "Return_lag_4", "Return_lag_5",
            "Volatility", "SMA_14", "EMA_14", "RSI", "MACD", "MACD_signal", "MACD_hist", "sentiment"
        ] + [col for col in data.columns if col.startswith("Sector_")]
        
        # Verify features
        missing_features = [f for f in features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing features: {missing_features}")
        
        print("Stock Recommendation System")
        print("==========================")
        
        while True:
            # Get user inputs
            sector, horizon, risk = get_user_inputs(data)
            
            # Get recommendations
            print(f"\nGenerating recommendations for {sector or 'All Sectors'}, {horizon} horizon, {risk} risk...")
            recommendations = get_recommendations(
                data,
                features,
                sector=sector,
                horizon=horizon,
                risk=risk,
                top_n=5
            )
            
            # Display results
            if not recommendations.empty:
                print("\nTop 5 Recommendations:")
                print(recommendations.to_string(index=False))
            else:
                print("No recommendations available.")
            
            # Ask to continue
            again = input("\nGenerate another recommendation? (yes/no): ").strip().lower()
            if again != "yes":
                print("Exiting.")
                break
        
    except Exception as e:
        print(f"Error: {e}")