# Stock Price Forecaster

A Streamlit app that estimates a stock's five trading day return and price from recent market data, technical indicators, and news sentiment. It also displays a stock overview, recent headlines, and an optional historical backtest.

## Models and methods

- A stacked LSTM regression model predicts the five trading day return from the previous 60 trading days of features.
- Inputs include OHLCV data, daily return, 10 and 30 day simple moving averages, a 10 day exponential moving average, 14 day RSI, Bollinger Band %B, 14 day ATR, volume change, daily range, and news sentiment features.
- Headline sentiment is scored with VADER. Headlines are collected from Yahoo Finance and Google News RSS; NewsAPI is also used when `NEWSAPI_KEY` is configured. A small built-in lexicon scorer is used if VADER cannot be loaded.
- Features are scaled with MinMaxScaler fitted only on the chronological training portion. The data is split by time, sequences are built from 60 days, and training uses early stopping and learning rate reduction.
- The displayed 95% price range uses 1.96 times the standard deviation of holdout residuals as an uncertainty estimate. It is a simple estimate, not a guarantee.

## Run locally

### Requirements

- Python 3.12
- Internet access for market data and news feeds

### Setup

Clone the repository and open a terminal in its root directory. Create and activate a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use `python3.12 -m venv .venv` and `source .venv/bin/activate` instead.

Install the pinned dependencies and start the app:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. Open it in a browser. Enter a ticker in the sidebar, select a prediction date, choose whether to show the backtest, and click **Run Forecast**. The app downloads recent market and news data and trains or refreshes a ticker's model artifacts when they are missing or out of date.

To use NewsAPI as an additional headline source, set `NEWSAPI_KEY` in your environment before starting the app. The application also uses Yahoo Finance and Google News RSS without that key.

The model can also be trained from the command line by running `python train_model.py`; its example configuration trains AAPL using a historical cutoff date.

## Online access

Setup to host the application online so it can be used through a link is currently in progress.

## Disclaimer

Forecasts are experimental model outputs for educational purposes, not financial advice. They can be wrong and should not be used as the sole basis for an investment decision.
