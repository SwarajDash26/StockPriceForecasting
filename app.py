import os
import json
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from utils.data_loader import get_stock_data
from utils.feature_engineer import add_technical_indicators
from utils.sentiment import (
    fetch_yfinance_headlines,
    score_headlines_vader,
    summarize_sentiment_until_date,
)

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Stock Price Forecaster (DL + Sentiment)", layout="wide")

N_STEPS = 60
FUTURE_DAYS = 5


# -----------------------------
# Helpers
# -----------------------------
def _artifact_paths(ticker: str):
    return {
        "model": f"models/{ticker}_lstm_model.keras",
        "scaler": f"models/{ticker}_scaler.pkl",
        "cols": f"models/{ticker}_feature_columns.json",
        "meta": f"models/{ticker}_error_std.json",
    }


def load_artifacts(ticker: str):
    paths = _artifact_paths(ticker)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts for this ticker:\n"
            + "\n".join([f"- {k}: {paths[k]}" for k in missing])
            + "\n\nTrain first, e.g.:\n"
            + f"  python -c \"from train_model import train_stock_model; train_stock_model('{ticker}', epochs=10)\""
        )

    model = load_model(paths["model"])
    scaler = joblib.load(paths["scaler"])

    with open(paths["cols"], "r", encoding="utf-8") as f:
        feature_columns = json.load(f)["feature_columns"]

    with open(paths["meta"], "r", encoding="utf-8") as f:
        meta = json.load(f)
    error_std = float(meta.get("error_std", 0.0))
    ci95 = float(meta.get("ci95", 0.0))

    return model, scaler, feature_columns, error_std, ci95


@st.cache_data(show_spinner=False)
def load_feature_dataframe(ticker: str, prediction_date: datetime.date) -> pd.DataFrame:
    raw = get_stock_data(ticker, prediction_date)
    feats = add_technical_indicators(raw)
    return feats


@st.cache_data(show_spinner=False)
def load_scored_headlines(ticker: str, max_items: int = 50) -> pd.DataFrame:
    h = fetch_yfinance_headlines(ticker, max_items=max_items)
    hs = score_headlines_vader(h)
    return hs


def predict_return(model, scaler, feature_columns, feats_df: pd.DataFrame) -> float:
    """
    Uses the last N_STEPS rows of feats_df to predict FUTURE_DAYS return.
    """
    if len(feats_df) < N_STEPS:
        raise ValueError(f"Not enough feature rows ({len(feats_df)}) to build a {N_STEPS}-step sequence.")

    # Enforce exact column order used in training
    missing = [c for c in feature_columns if c not in feats_df.columns]
    if missing:
        raise ValueError(f"Feature columns missing at inference: {missing}")

    latest = feats_df[feature_columns].iloc[-N_STEPS:].copy()
    scaled = scaler.transform(latest.values)
    seq = scaled.reshape(1, N_STEPS, len(feature_columns))

    pred = float(model.predict(seq, verbose=0)[0][0])
    return pred


def price_from_return(current_price: float, predicted_return: float) -> float:
    return float(current_price * (1.0 + predicted_return))


# -----------------------------
# UI
# -----------------------------
st.title("Stock Price Forecaster (Deep Learning + Sentiment)")
st.caption(
    "Predicts the *5-day return* using an LSTM trained on OHLCV + core technical indicators. "
    "Adds a simple news headline sentiment layer for a demo-friendly hybrid forecast."
)

st.sidebar.header("Inputs")

ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()

# Default prediction date: 10 days ago (helps ensure future data exists for backtest)
prediction_date = st.sidebar.date_input(
    "Prediction Date",
    datetime.now().date() - timedelta(days=10),
    max_value=datetime.now().date() - timedelta(days=5),
)

enable_backtest = st.sidebar.checkbox("Enable backtest (compare to actual close 5 business days later)", value=True)

st.sidebar.subheader("Sentiment Settings")
sentiment_days_back = st.sidebar.slider("Lookback window for headlines (days)", 1, 14, 7)
sentiment_alpha = st.sidebar.slider(
    "Sentiment influence (alpha)",
    0.0, 0.10, 0.02, 0.005,
    help="Adjusted return = LSTM_return + alpha * mean_compound_sentiment (compound in [-1,1])",
)

run_btn = st.sidebar.button("Run Forecast")


if run_btn:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    try:
        with st.spinner("Loading model artifacts..."):
            model, scaler, feature_columns, error_std, ci95 = load_artifacts(ticker)

        with st.spinner("Loading price data and engineering features..."):
            feats = load_feature_dataframe(ticker, prediction_date)

        with st.spinner("Predicting with LSTM..."):
            pred_return = predict_return(model, scaler, feature_columns, feats)

        # Current price (last close in features df)
        current_close = float(feats["Close"].iloc[-1])
        pred_price = price_from_return(current_close, pred_return)

        # Confidence band (simple residual-based proxy from training holdout)
        lower_price = price_from_return(current_close, pred_return - ci95)
        upper_price = price_from_return(current_close, pred_return + ci95)

        # --- Sentiment ---
        with st.spinner("Fetching headlines and scoring sentiment..."):
            headlines_scored = load_scored_headlines(ticker, max_items=60)
            sent_summary = summarize_sentiment_until_date(
                headlines_scored, prediction_date, max_days_back=sentiment_days_back
            )

        adjusted_return = pred_return + sentiment_alpha * sent_summary.mean_compound
        adjusted_price = price_from_return(current_close, adjusted_return)

        # -----------------------------
        # Results
        # -----------------------------
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Close", f"${current_close:,.2f}")
        c2.metric("LSTM Predicted 5D Return", f"{pred_return:+.2%}")
        c3.metric("LSTM Predicted Price (5D)", f"${pred_price:,.2f}")
        c4.metric("95% Range (Price)", f"${lower_price:,.2f} – ${upper_price:,.2f}")

        st.subheader("Sentiment (News Headlines)")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Headline Count", f"{sent_summary.headline_count}")
        s2.metric("Mean Compound", f"{sent_summary.mean_compound:+.3f}")
        s3.metric("Pos / Neu / Neg", f"{sent_summary.pos_frac:.0%} / {sent_summary.neu_frac:.0%} / {sent_summary.neg_frac:.0%}")
        s4.metric("Sentiment-Adjusted Price (5D)", f"${adjusted_price:,.2f}", delta=f"{adjusted_return:+.2%}")

        if sent_summary.headline_count == 0:
            st.info("No headlines found in the lookback window (or yfinance returned none). Sentiment adjustment will be ~0.")

        # Show headlines table
        if len(headlines_scored) > 0:
            st.write("Recent headlines (scored):")
            show_df = headlines_scored.copy()
            show_df["published_at"] = pd.to_datetime(show_df["published_at"]).dt.strftime("%Y-%m-%d %H:%M UTC")
            st.dataframe(show_df[["published_at", "source", "compound", "title"]].head(15), use_container_width=True)

        # Price chart
        st.subheader("Price History + Forecast Markers")
        fig, ax = plt.subplots(figsize=(12, 5))
        recent = feats.tail(120)
        ax.plot(recent.index, recent["Close"], label="Close", linewidth=2)
        ax.axhline(current_close, linestyle="--", color="gray", alpha=0.7, label="Current Close")
        ax.axhline(pred_price, linestyle="--", color="green", alpha=0.8, label="Predicted Price (LSTM)")
        ax.axhline(adjusted_price, linestyle="--", color="blue", alpha=0.8, label="Predicted Price (Adj)")

        ax.set_title(f"{ticker} Close Price (recent) and 5D Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        # Backtest (optional)
        if enable_backtest:
            st.subheader("Backtest (single point)")
            future_date = pd.to_datetime(prediction_date) + pd.tseries.offsets.BDay(FUTURE_DAYS)
            try:
                actual_df = get_stock_data(ticker, future_date.date())
                actual_close = float(actual_df["Close"].iloc[-1])

                err_lstm = abs(pred_price - actual_close)
                err_adj = abs(adjusted_price - actual_close)

                b1, b2, b3 = st.columns(3)
                b1.metric(f"Actual Close on {future_date.date()}", f"${actual_close:,.2f}")
                b2.metric("Abs Error (LSTM)", f"${err_lstm:,.2f}")
                b3.metric("Abs Error (Adj)", f"${err_adj:,.2f}")
            except Exception as e:
                st.warning(f"Backtest failed (often due to missing future data): {e}")

        st.caption(
            "Notes: This is a prototype. Sentiment is computed from recent headlines and applied as a simple adjustment. "
            "For a stronger approach, we would train on historical sentiment features aligned by date."
        )

    except Exception as e:
        st.error(str(e))
        st.stop()


st.sidebar.markdown("---")
st.sidebar.info(
    "Workflow:\n"
    "1) Train: python train_model.py (or train_stock_model(...) )\n"
    "2) Run app: streamlit run app.py\n"
    "3) Forecast: loads saved artifacts per ticker\n"
)