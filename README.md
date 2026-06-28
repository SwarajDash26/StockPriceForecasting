# Stock Price Forecaster

A Streamlit application that forecasts a stock's five-business-day move using
an LSTM trained on historical prices, technical indicators, and recent news
sentiment.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Create an app at [share.streamlit.io](https://share.streamlit.io) with:

- Repository: `SwarajDash26/StockPriceForecasting`
- Branch: the branch containing the deployment commit
- Main file path: `app.py`
- Python: `3.12`

No secret is required. If a `NEWSAPI_KEY` environment variable is provided,
the app can use NewsAPI in addition to its keyless news sources.

## Important note

Forecasts are experimental model outputs for educational purposes and are not
financial advice. They can be wrong and should not be used as the sole basis
for an investment decision.
