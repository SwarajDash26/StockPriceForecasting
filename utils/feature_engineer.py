import pandas as pd
import numpy as np


def _standardize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with standardized OHLCV columns:
    ['Open', 'High', 'Low', 'Close', 'Volume']

    Handles MultiIndex columns and common variations.
    """
    out = df.copy()

    # Flatten MultiIndex columns if present
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [c[0] for c in out.columns]

    # Try to find columns by name (case-insensitive)
    cols = {c.lower(): c for c in out.columns}

    def find_col(keys):
        for k in keys:
            for col_lower, original in cols.items():
                if k in col_lower:
                    return original
        return None

    close_col = find_col(["close"])
    open_col = find_col(["open"])
    high_col = find_col(["high"])
    low_col = find_col(["low"])
    volume_col = find_col(["volume"])

    # Basic validation
    missing = []
    if close_col is None:
        missing.append("Close")
    if high_col is None:
        missing.append("High")
    if low_col is None:
        missing.append("Low")
    if open_col is None:
        missing.append("Open")
    if volume_col is None:
        missing.append("Volume")

    if missing:
        raise ValueError(
            f"Missing required OHLCV columns: {missing}. "
            f"Available columns: {list(out.columns)}"
        )

    # Rename to standard
    out = out.rename(
        columns={
            open_col: "Open",
            high_col: "High",
            low_col: "Low",
            close_col: "Close",
            volume_col: "Volume",
        }
    )

    # Ensure datetime index sorted
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    return out


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simplified, core set of technical indicators.

    Output columns (stable order):
      Open, High, Low, Close, Volume,
      Return_1D,
      SMA_10, SMA_30,
      EMA_10,
      RSI_14,
      BBP_20,
      ATR_14,
      Vol_Change,
      Range_Pct

    Drops rows with NaNs caused by rolling windows.
    """
    d = _standardize_ohlcv_columns(df)

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    # 1) Simple returns
    d["Return_1D"] = close.pct_change()

    # 2) Trend
    d["SMA_10"] = close.rolling(window=10).mean()
    d["SMA_30"] = close.rolling(window=30).mean()
    d["EMA_10"] = close.ewm(span=10, adjust=False).mean()

    # 3) RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    d["RSI_14"] = 100 - (100 / (1 + rs))

    # 4) Bollinger %B (20)
    sma_20 = close.rolling(window=20).mean()
    std_20 = close.rolling(window=20).std()
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    d["BBP_20"] = (close - bb_lower) / (bb_upper - bb_lower)

    # 5) ATR (14)
    high_low = high - low
    high_close = (high - close.shift()).abs()
    low_close = (low - close.shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR_14"] = true_range.rolling(window=14).mean()

    # 6) Volume change
    d["Vol_Change"] = volume.pct_change()

    # 7) Daily range as % of close
    d["Range_Pct"] = (high - low) / close

    # Keep stable column order
    final_cols = [
        "Open", "High", "Low", "Close", "Volume",
        "Return_1D",
        "SMA_10", "SMA_30",
        "EMA_10",
        "RSI_14",
        "BBP_20",
        "ATR_14",
        "Vol_Change",
        "Range_Pct",
    ]
    d = d[final_cols]

    # Drop NaNs from rolling calculations
    d = d.dropna()

    # A small safety check (helps catch weird downloads)
    if len(d) < 100:
        raise ValueError(f"Not enough rows after indicator generation: {len(d)} rows remain.")

    return d


if __name__ == "__main__":
    from utils.data_loader import get_stock_data

    test_date = pd.to_datetime("2023-12-01").date()
    raw = get_stock_data("AAPL", test_date)
    feats = add_technical_indicators(raw)
    print("Feature engineering OK.")
    print(feats.tail())
    print("Columns:", feats.columns.tolist())