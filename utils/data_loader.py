import yfinance as yf
import pandas as pd

def get_stock_data(ticker, prediction_date):
    """
    Downloads 3 years of historical stock data leading up to a specified prediction date.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL').
    prediction_date (datetime.date): The date from which to predict forward.

    Returns:
    pd.DataFrame: A DataFrame with OHLCV data from (prediction_date - 3 years) to (prediction_date).
    """

    # Calculate the start date as 3 years before the prediction date
    start_date = prediction_date - pd.DateOffset(years=3)
    # Format dates for yfinance
    start_str = start_date.strftime('%Y-%m-%d')
    # We need data up to and including the prediction date.
    # Add 5 days to the prediction date to ensure we get it, then we'll trim later.
    end_date = prediction_date + pd.DateOffset(days=5)
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        # Download the data with explicit parameters to avoid warnings
        # Use auto_adjust=True to get simpler column names
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError(f"No data found for ticker '{ticker}'. Please check the symbol.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        # Rename common variations to standard names
        rename_map = {
            "Adj Close": "Close",
            "Close": "Close",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Volume": "Volume"
        }
        df = df.rename(columns=rename_map)

        # Ensure the index is a DateTimeIndex
        df.index = pd.to_datetime(df.index)
        # Filter to only include dates up to the prediction date
        df = df[df.index.date <= prediction_date]

        # Check if we have enough data
        if len(df) < 60:  # Arbitrary minimum, e.g., 3 months
            raise ValueError(f"Not enough data for {ticker} on {prediction_date}. Only {len(df)} days available.")
        
        # Add the ticker name to the DataFrame for reference
        df.name = ticker
        
        return df

    except Exception as e:
        raise Exception(f"An error occurred while downloading data: {str(e)}")

# Example usage for testing:
if __name__ == "__main__":
    # Test with a specific date
    test_date = pd.to_datetime('2023-12-01').date() # Predict from Dec 1, 2023
    data = get_stock_data('AAPL', test_date)
    print("Data downloaded successfully!")
    print(data.tail())
    print(f"\nData shape: {data.shape}")
    print(f"Date Range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Columns: {data.columns.tolist()}")