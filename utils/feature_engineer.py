import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """
    Adds technical indicators to the DataFrame as new columns.
    This function manually calculates indicators for compatibility.

    Parameters:
    df (pd.DataFrame): DataFrame with OHLCV data.

    Returns:
    pd.DataFrame: The original DataFrame with new technical indicator columns.
    """

    # Make a copy of the DataFrame to avoid modifying the original
    df = df.copy()

    # Handle MultiIndex columns by flattening them first
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip('_') for col in df.columns]
    
    # Dynamically find the OHLCV columns
    close_col = None
    high_col = None
    low_col = None
    open_col = None
    volume_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'close' in col_lower and 'adj' not in col_lower:
            close_col = col
        elif 'high' in col_lower:
            high_col = col
        elif 'low' in col_lower:
            low_col = col
        elif 'open' in col_lower:
            open_col = col
        elif 'volume' in col_lower:
            volume_col = col
    
    # Use default names if not found
    if close_col is None:
        close_col = 'Close'
    if high_col is None:
        high_col = 'High'
    if low_col is None:
        low_col = 'Low'
    if open_col is None:
        open_col = 'Open'
    if volume_col is None:
        volume_col = 'Volume'

    # 1. Trend Indicators
    # Simple Moving Averages
    df['SMA_10'] = df[close_col].rolling(window=10).mean()
    df['SMA_50'] = df[close_col].rolling(window=50).mean()

    # 2. Momentum Indicators
    # Relative Strength Index (RSI)
    delta = df[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # 3. Volatility Indicators
    # Bollinger Bands
    sma_20 = df[close_col].rolling(window=20).mean()
    std_20 = df[close_col].rolling(window=20).std()
    bb_upper = sma_20 + (std_20 * 2)
    bb_lower = sma_20 - (std_20 * 2)
    df['BB_Upper'] = bb_upper
    df['BB_Lower'] = bb_lower
    df['BBP_20'] = (df[close_col] - bb_lower) / (bb_upper - bb_lower)  # %B

    # Average True Range (ATR)
    high_low = df[high_col] - df[low_col]
    high_close = np.abs(df[high_col] - df[close_col].shift())
    low_close = np.abs(df[low_col] - df[close_col].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # 4. Volume-based Indicator
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df[close_col].diff()) * df[volume_col]).fillna(0).cumsum()

    # 5. Create a simple price-based feature: Daily Return
    df['Daily_Return'] = df[close_col].pct_change()

    # After adding many indicators, the first ~50 rows will be NaN (due to calculation windows)
    # Drop all rows with any NaN values to have a clean dataset for the model
    df.dropna(inplace=True)

    # Ensure we have exactly the expected columns in the right order
    expected_columns = [
        close_col, high_col, low_col, open_col, volume_col,
        'SMA_10', 'SMA_50', 'RSI_14', 'BB_Upper', 'BB_Lower', 
        'BBP_20', 'ATR_14', 'OBV', 'Daily_Return'
    ]
    
    # Keep only the expected columns (in case some were missing)
    df = df[expected_columns]
    
    # Rename the OHLCV columns to standard names for consistency
    df = df.rename(columns={
        close_col: 'Close',
        high_col: 'High', 
        low_col: 'Low',
        open_col: 'Open',
        volume_col: 'Volume'
    })

    return df

# Example usage for testing:
if __name__ == "__main__":
    # This block only runs if you run this file directly, not when imported
    from data_loader import get_stock_data

    # Get some sample data
    test_date = pd.to_datetime('2023-12-01').date()
    sample_data = get_stock_data('AAPL', test_date)
    print(f"Original data shape: {sample_data.shape}")
    print(f"Original columns: {sample_data.columns.tolist()}")

    # Add technical indicators
    featured_data = add_technical_indicators(sample_data)

    # Inspect the results
    print(f"New data shape after adding features: {featured_data.shape}")
    print("\nFirst 3 rows of the new technical indicator columns:")
    # Show only the first 3 rows of the new indicator columns (not the original OHLCV)
    new_columns = [col for col in featured_data.columns if col not in sample_data.columns]
    print(featured_data[new_columns].head(3))