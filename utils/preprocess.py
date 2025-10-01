import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_target(df, future_days=5):
    """
    Creates the target variable for our REGRESSION problem.
    We predict the percentage change in 'future_days' days.

    Parameters:
    df (pd.DataFrame): DataFrame with technical indicators.
    future_days (int): Number of days ahead to predict.

    Returns:
    pd.DataFrame: DataFrame with the new 'Target' column (percentage change).
    """
    df = df.copy()
    
    # First, let's simplify the MultiIndex columns to single-level columns
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the MultiIndex columns
        df.columns = ['_'.join(col).strip('_') for col in df.columns]
    
    # Dynamically find the close column
    close_col = None
    for col in df.columns:
        if 'close' in col.lower() and 'adj' not in col.lower():
            close_col = col
            break
    
    if close_col is None:
        # If no close column found, try any column with 'close'
        for col in df.columns:
            if 'close' in col.lower():
                close_col = col
                break
    
    if close_col is None:
        raise ValueError("Could not find a close price column in the DataFrame")
    
    # Calculate future price: shift the close price backward by 'future_days'
    future_price = df[close_col].shift(-future_days)
    
    # Calculate PERCENTAGE CHANGE from current to future price (REGRESSION TARGET)
    df['Target'] = (future_price - df[close_col]) / df[close_col]
    
    # Drop the last 'future_days' rows which will have NaN for the target
    df = df.dropna(subset=['Target'])
    
    return df
    
def prepare_data_for_lstm(df, n_steps=60, test_size=0.2):
    """
    Prepares the data for LSTM training by scaling and creating sequences.

    Parameters:
    df (pd.DataFrame): DataFrame with features and target.
    n_steps (int): Number of past days to use for prediction (sequence length).
    test_size (float): Proportion of data to use for testing.

    Returns:
    tuple: X_train, X_test, y_train, y_test, feature_scaler
    """
    # Separate features and target
    # Make sure we only keep numeric columns for features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Target' in numeric_cols:
        numeric_cols.remove('Target')
    
    features = df[numeric_cols]
    target = df['Target']
    
    # 1. Scale the features
    feature_scaler = MinMaxScaler()
    scaled_features = feature_scaler.fit_transform(features)
    
    # 2. Create sequences for LSTM
    X, y = [], []
    for i in range(n_steps, len(scaled_features)):
        # Get sequence of n_steps days
        X.append(scaled_features[i-n_steps:i])
        # Get target for the current day (which corresponds to future prediction)
        y.append(target.iloc[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # 3. Split into train and test sets CHRONOLOGICALLY
    split_index = int(len(X) * (1 - test_size))
    
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, feature_scaler

# Let's add some debug information to see what's happening
if __name__ == "__main__":
    from data_loader import get_stock_data
    from feature_engineer import add_technical_indicators
    
    # Get and prepare the data
    test_date = pd.to_datetime('2023-12-01').date()
    data = get_stock_data('AAPL', test_date)
    print(f"Raw data shape: {data.shape}")
    
    featured_data = add_technical_indicators(data)
    print(f"Featured data shape: {featured_data.shape}")
    print(f"Featured data columns: {featured_data.columns.tolist()}")
    
    # Create target variable
    data_with_target = create_target(featured_data, future_days=5)
    print(f"Data shape after adding target: {data_with_target.shape}")
    
    if 'Target' in data_with_target.columns:
        print(f"Target value counts:\n{data_with_target['Target'].value_counts()}")
        print(f"Target NaN values: {data_with_target['Target'].isna().sum()}")
    else:
        print("Target column was not created successfully")
        print(f"Available columns: {data_with_target.columns.tolist()}")
    
    # Prepare for LSTM only if target was created
    if 'Target' in data_with_target.columns:
        X_train, X_test, y_train, y_test, scaler = prepare_data_for_lstm(data_with_target)
        print("\nData preparation completed successfully!")
        print(f"Number of training sequences: {len(X_train)}")
        print(f"Number of testing sequences: {len(X_test)}")