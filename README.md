Stock Price Forecasting with LSTM

This project implements an end-to-end deep learning pipeline for predicting stock price movements using a stacked LSTM model enriched with technical indicators.
It includes data collection, feature engineering, model training, evaluation, and an interactive Streamlit dashboard for making forecasts and backtesting against historical data.

Features:

Data pipeline using Yahoo Finance API (historical OHLCV data).

Feature engineering with RSI, SMA, Bollinger Bands, ATR, and OBV.

Stacked LSTM network for time series forecasting with uncertainty estimation.

Per-ticker model saving/loading for dynamic predictions.

Streamlit dashboard with ticker/date input and interactive prediction results.

Backtesting mode to compare predictions with actual outcomes.

Model Performance: 

Achieved MAE of 2–4% on test data across multiple tickers.

>93% of actuals fell within model’s 95% confidence intervals.

Demonstrated consistent performance and generalization on MSFT, NVDA, AAPL.

Errors remained balanced (train vs test), showing low overfitting.

Learning Outcomes:

Gained hands-on experience in time series forecasting with deep learning.

Combined finance domain knowledge (technical indicators) with ML modeling.

Implemented model evaluation & uncertainty calibration.

Learned to deploy a data science app for interactive use.

📂 Project Structure
├── app.py              # Streamlit dashboard
├── data_loader.py      # Data ingestion from Yahoo Finance
├── feature_engineer.py # Technical indicator engineering
├── preprocess.py       # Scaling, target creation, sequence prep
├── model_builder.py    # Stacked LSTM model definition
├── train_model.py      # Training pipeline + evaluation
├── models/             # Saved models (per ticker) + scalers
└── README.md           # Project documentation

Installation & Usage:
1. Clone the repo
git clone https://github.com/SwarajDash26/StockPriceForecasting.git
cd StockPriceForecasting

2. Install Dependencies

3. Run the Streamlit dashboard
streamlit run app.py

4. Using the app

Enter a stock ticker (e.g., MSFT, NVDA, AAPL).

Select a prediction anchor date.

App predicts 5 days forward with confidence bands.
