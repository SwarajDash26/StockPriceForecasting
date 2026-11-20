import pandas as pd
import yfinance as yf


MIN_HISTORY_ROWS = 60


def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    rename_map = {
        "Adj Close": "Close",
        "Close": "Close",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    df.index = pd.to_datetime(df.index)
    return df


def _download_with_fallbacks(ticker: str, start_str: str, end_str: str) -> pd.DataFrame:
    normalized_ticker = str(ticker).upper().strip()
    errors = []

    try:
        df = yf.download(
            normalized_ticker,
            start=start_str,
            end=end_str,
            progress=False,
            auto_adjust=True,
            threads=False,
        )
        if df is not None and not df.empty:
            return _normalize_ohlcv_columns(df)
        errors.append("yf.download returned no rows")
    except Exception as exc:
        errors.append(f"yf.download failed: {exc}")

    try:
        ticker_obj = yf.Ticker(normalized_ticker)
        df = ticker_obj.history(
            start=start_str,
            end=end_str,
            auto_adjust=True,
        )
        if df is not None and not df.empty:
            return _normalize_ohlcv_columns(df)
        errors.append("Ticker.history returned no rows")
    except Exception as exc:
        errors.append(f"Ticker.history failed: {exc}")

    raise ValueError(
        f"No price history was returned for '{normalized_ticker}' between {start_str} and {end_str}. "
        f"Yahoo Finance may be temporarily unavailable, the market may have insufficient history for that window, "
        f"or the ticker may be unsupported. Details: {' | '.join(errors)}"
    )

def get_stock_data(ticker, prediction_date):
    """
    Downloads 3 years of historical stock data leading up to a specified prediction date.

    Parameters:
    ticker (str): The stock ticker symbol (e.g., 'AAPL').
    prediction_date (datetime.date): The date from which to predict forward.

    Returns:
    pd.DataFrame: A DataFrame with OHLCV data from (prediction_date - 3 years) to (prediction_date).
    """

    normalized_ticker = str(ticker).upper().strip()
    prediction_date = pd.to_datetime(prediction_date).date()

    start_date = pd.Timestamp(prediction_date) - pd.DateOffset(years=3)
    start_str = start_date.strftime('%Y-%m-%d')
    end_date = pd.Timestamp(prediction_date) + pd.DateOffset(days=5)
    end_str = end_date.strftime('%Y-%m-%d')

    try:
        df = _download_with_fallbacks(normalized_ticker, start_str, end_str)
        df = df[df.index.date <= prediction_date]

        if len(df) < MIN_HISTORY_ROWS:
            raise ValueError(
                f"Not enough data for {normalized_ticker} on {prediction_date}. "
                f"Only {len(df)} trading days were available; at least {MIN_HISTORY_ROWS} are required."
            )
        
        df.name = normalized_ticker
        
        return df

    except Exception as e:
        raise Exception(f"An error occurred while downloading data: {str(e)}")

if __name__ == "__main__":
    test_date = pd.to_datetime('2023-12-01').date()
    data = get_stock_data('AAPL', test_date)
    print("Data downloaded successfully!")
    print(data.tail())
    print(f"\nData shape: {data.shape}")
    print(f"Date Range: {data.index.min().date()} to {data.index.max().date()}")
    print(f"Columns: {data.columns.tolist()}")
