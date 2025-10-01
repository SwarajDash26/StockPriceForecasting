import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
from tensorflow.keras.models import load_model

# Import your utility functions
from utils.data_loader import get_stock_data
from utils.feature_engineer import add_technical_indicators
from utils.preprocess import create_target, prepare_data_for_lstm

# Page configuration
st.set_page_config(
    page_title="Stock Price Forecaster",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Price Prediction with LSTM")
st.markdown("""
This app predicts stock price movements 5 days into the future using a deep learning LSTM model.
Select a stock ticker and date to get started!
""")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Stock ticker input
ticker = st.sidebar.text_input("Stock Ticker Symbol", "AAPL").upper()

enable_backtest = st.sidebar.checkbox("Enable Backtesting (compare with actual 5-day future price)")

# Date input with reasonable defaults
end_date = st.sidebar.date_input(
    "Prediction Date",
    datetime.now() - timedelta(days=10),  # Default to 10 days ago to ensure data availability
    max_value=datetime.now() - timedelta(days=5)  # Can't predict today since we need latest data
)

# Main processing function
def make_prediction(ticker, prediction_date):
    """Main function to load data, process, train if needed, and make prediction"""

    # Display progress
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("Loading historical data...")
        data = get_stock_data(ticker, prediction_date)
        progress_bar.progress(20)

        status_text.text("Engineering features...")
        featured_data = add_technical_indicators(data)
        progress_bar.progress(40)

        status_text.text("Creating prediction target...")
        data_with_target = create_target(featured_data, future_days=5)
        progress_bar.progress(60)

        # Try loading pre-trained model and scaler
        try:
            model = load_model(f'models/{ticker}_lstm_model.keras')
            scaler = joblib.load(f'models/{ticker}_scaler.pkl')
            status_text.text(f"Loaded pre-trained model for {ticker}")
        except:
            # If missing → train dynamically
            st.warning(f"No pre-trained model found for {ticker}. Training a new one now...")

            from train_model import train_stock_model
            with st.spinner(f"Training LSTM model for {ticker}... This may take a few minutes"):
                model, history, test_mae = train_stock_model(
                    ticker=ticker,
                    prediction_date=prediction_date,
                    future_days=5,
                    n_steps=60
                )

            # Reload scaler after training
            scaler = joblib.load(f'models/{ticker}_scaler.pkl')
            status_text.text(f"Training complete. Model ready for {ticker}!")

        # Prepare the most recent sequence for prediction
        latest_features = featured_data.iloc[-60:].copy()  # Last 60 days
        scaled_features = scaler.transform(latest_features)
        sequence = scaled_features.reshape(1, 60, -1)  # Reshape for LSTM

        # Make prediction
        status_text.text("Making prediction...")
        predicted_pct_change = model.predict(sequence, verbose=0)[0][0]

        progress_bar.progress(100)
        status_text.text("Prediction complete!")

        return data, predicted_pct_change, latest_features

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None



        # Prepare the most recent sequence for prediction
        latest_features = featured_data.iloc[-60:].copy()  # Last 60 days
        scaled_features = scaler.transform(latest_features)
        sequence = scaled_features.reshape(1, 60, -1)  # Reshape for LSTM
        
        # Make prediction
        status_text.text("Making prediction...")
        predicted_pct_change = model.predict(sequence, verbose=0)[0][0]
        
        progress_bar.progress(100)
        status_text.text("Prediction complete!")
        
        return data, predicted_pct_change, latest_features
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None, None, None

# Display results function
def display_results(ticker, prediction_date, data, predicted_pct_change, latest_features, enable_backtest=False):
    """Display the prediction results beautifully"""
    
    # For auto_adjust=True data, we get simple column names: ['Open', 'High', 'Low', 'Close', 'Volume']
    close_col = 'Close'
    
    if close_col not in data.columns:
        st.error(f"Could not find 'Close' column in the data. Available columns: {list(data.columns)}")
        return
    
    latest_close = data[close_col].iloc[-1]
    
    # Calculate predicted price and confidence interval
    error_std = 0.0305  # From your training results
    confidence_interval = 1.96 * error_std
    
    predicted_price = latest_close * (1 + predicted_pct_change)
    lower_bound = latest_close * (1 + predicted_pct_change - confidence_interval)
    upper_bound = latest_close * (1 + predicted_pct_change + confidence_interval)
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Price", f"${latest_close:.2f}")
    
    with col2:
        st.metric("Predicted Price (5 days)", f"${predicted_price:.2f}", 
                 delta=f"{predicted_pct_change:.2%}")
    
    with col3:
        st.metric("95% Confidence Range", 
                 f"${lower_bound:.2f} - ${upper_bound:.2f}")
    
    if enable_backtest:
        st.subheader("🔎 Backtesting Results")

        try:
            from utils.data_loader import get_stock_data
            import pandas as pd

            # Get actual price 5 trading days later
            future_date = pd.to_datetime(prediction_date) + pd.tseries.offsets.BDay(5)
            actual_data = get_stock_data(ticker, future_date.date())
            actual_close = actual_data[close_col].iloc[-1]

            mae = abs(predicted_price - actual_close)
            pct_error = (mae / actual_close) * 100

            st.write(f"**Actual closing price on {future_date.date()}:** ${actual_close:.2f}")
            st.write(f"**Prediction Error:** ${mae:.2f} ({pct_error:.2f}%)")

            if pct_error < 2:
                st.success("✅ Prediction was very close!")
            else:
                st.warning("⚠️ Prediction deviated more than 2%.")

        except Exception as e:
            st.error(f"Backtesting failed: {e}") 
    # Price chart
    st.subheader("Price History")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot historical price (last 90 days for better visualization)
    recent_data = data.iloc[-90:]
    ax.plot(recent_data.index, recent_data[close_col], label='Historical Price', linewidth=2)
    
    # Add current and predicted price lines
    ax.axhline(y=latest_close, color='r', linestyle='--', alpha=0.7, label='Current Price')
    ax.axhline(y=predicted_price, color='g', linestyle='--', alpha=0.7, label='Predicted Price')
    
    # Add confidence interval area
    last_date = data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 6)]
    if len(future_dates) > 0:
        ax.fill_between(future_dates, 
                       [lower_bound] * len(future_dates), 
                       [upper_bound] * len(future_dates), 
                       color='orange', alpha=0.2, label='95% CI')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Price ($)')
    ax.set_title(f'{ticker} Price Prediction')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    
    # Interpretation
    st.subheader("Interpretation")
    if predicted_pct_change > 0:
        st.success(f"📈 Bullish Prediction: {ticker} is predicted to increase by {predicted_pct_change:.2%} over the next 5 trading days.")
    else:
        st.warning(f"📉 Bearish Prediction: {ticker} is predicted to decrease by {abs(predicted_pct_change):.2%} over the next 5 trading days.")
    
    st.info(f"**Confidence Level:** There's a 95% probability that the actual price will be between ${lower_bound:.2f} and ${upper_bound:.2f}.")
    
# Run the app
if st.sidebar.button("Run Prediction"):
    if not ticker:
        st.error("Please enter a stock ticker symbol.")
    else:
        with st.spinner("Crunching numbers... This may take a minute"):
            data, predicted_pct_change, latest_features = make_prediction(ticker, end_date)

            if data is not None and predicted_pct_change is not None:
                display_results(ticker, end_date, data, predicted_pct_change, latest_features, enable_backtest)

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **How it works:**
    - Uses LSTM neural network trained on 3 years of historical data
    - Analyzes technical indicators (RSI, SMA, Bollinger Bands, etc.)
    - Provides 5-day price predictions with confidence intervals
    """
)