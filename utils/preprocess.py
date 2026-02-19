import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def create_target(df: pd.DataFrame, future_days: int = 5) -> pd.DataFrame:
    """
    Creates the regression target:
    Target = (Close[t+future_days] - Close[t]) / Close[t]

    Notes:
    - Assumes a 'Close' column exists (your feature_engineer standardizes to this).
    - Drops the last `future_days` rows (no future price available).
    """
    df = df.copy()

    # Handle MultiIndex columns just in case
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip("_") for col in df.columns]

    if "Close" not in df.columns:
        # Fallback: try to find a close-like column
        close_col = None
        for col in df.columns:
            c = col.lower()
            if "close" in c and "adj" not in c:
                close_col = col
                break
        if close_col is None:
            raise ValueError("Could not find a 'Close' column in the DataFrame.")
        df = df.rename(columns={close_col: "Close"})

    future_close = df["Close"].shift(-future_days)
    df["Target"] = (future_close - df["Close"]) / df["Close"]

    # Drop rows where target is NaN (last future_days)
    df = df.dropna(subset=["Target"])
    return df


def prepare_data_for_lstm(
    df: pd.DataFrame,
    n_steps: int = 60,
    test_size: float = 0.2,
):
    """
    Prepares data for LSTM training:
    - Uses numeric columns except 'Target' as features
    - Chronological split (no shuffling)
    - Fits scaler ONLY on training feature rows (prevents leakage)
    - Builds sequences of length n_steps

    Returns:
      X_train, X_test, y_train, y_test, feature_scaler, feature_columns
    """
    if "Target" not in df.columns:
        raise ValueError("DataFrame must contain a 'Target' column. Call create_target(...) first.")

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    df = df.copy()

    # Features: numeric columns except Target
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Target" in numeric_cols:
        numeric_cols.remove("Target")

    if len(numeric_cols) == 0:
        raise ValueError("No numeric feature columns found.")

    feature_columns = numeric_cols
    features = df[feature_columns]
    target = df["Target"].astype(float)

    if len(df) <= n_steps + 5:
        raise ValueError(f"Not enough rows ({len(df)}) to create sequences with n_steps={n_steps}.")

    # Chronological split point in ROW space (not sequence space)
    split_row = int(len(df) * (1 - test_size))

    # We need split_row > n_steps so there is at least one training sequence
    if split_row <= n_steps:
        raise ValueError(
            f"Split produces no training sequences. "
            f"len(df)={len(df)}, n_steps={n_steps}, split_row={split_row}. "
            f"Reduce n_steps or reduce test_size."
        )

    # Fit scaler on training portion ONLY (prevents leakage)
    feature_scaler = MinMaxScaler()
    feature_scaler.fit(features.iloc[:split_row])

    scaled_features = feature_scaler.transform(features)

    # Build sequences
    X, y = [], []
    # i is the "current" row index (sequence ends at i-1, predicts target at i)
    for i in range(n_steps, len(scaled_features)):
        X.append(scaled_features[i - n_steps : i])
        y.append(target.iloc[i])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Convert split from row space to sequence space:
    # sequence index j corresponds to row i = n_steps + j
    train_seq_count = split_row - n_steps

    X_train = X[:train_seq_count]
    y_train = y[:train_seq_count]
    X_test = X[train_seq_count:]
    y_test = y[train_seq_count:]

    print(f"Feature columns ({len(feature_columns)}): {feature_columns}")
    print(f"Split row: {split_row} of {len(df)} rows")
    print(f"X_train shape: {X_train.shape} | y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape} | y_test shape:  {y_test.shape}")

    return X_train, X_test, y_train, y_test, feature_scaler, feature_columns


if __name__ == "__main__":
    # Quick self-test (optional)
    from utils.data_loader import get_stock_data
    from utils.feature_engineer import add_technical_indicators

    test_date = pd.to_datetime("2023-12-01").date()
    raw = get_stock_data("AAPL", test_date)
    feats = add_technical_indicators(raw)
    with_target = create_target(feats, future_days=5)

    X_train, X_test, y_train, y_test, scaler, cols = prepare_data_for_lstm(
        with_target, n_steps=60, test_size=0.2
    )
    print("Preprocess OK.")