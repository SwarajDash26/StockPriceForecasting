import json
import os
from html import escape
from datetime import date, datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import yfinance as yf
from tensorflow.keras.models import load_model

from train_model import train_stock_model
from utils.data_loader import get_stock_data
from utils.feature_engineer import add_technical_indicators
from utils.sentiment import (
    build_daily_sentiment_features,
    summarize_sentiment_until_date,
)

# ----
# Config
# ----
st.set_page_config(page_title="Stock Price Forecaster (DL + Sentiment)", layout="wide")

N_STEPS = 60
FUTURE_DAYS = 5
EPOCHS = 12
SENTIMENT_LOOKBACK_DAYS = 10
SENTIMENT_PULL_DAYS = 30
ARTIFACT_DEPENDENCY_FILES = (
    "train_model.py",
    "utils/data_loader.py",
    "utils/feature_engineer.py",
    "utils/model_builder.py",
    "utils/preprocess.py",
    "utils/sentiment.py",
)
ARTIFACT_MTIME_GRACE_SECONDS = 60


# ----
# Helpers
# ----
def _artifact_paths(ticker: str):
    return {
        "model": f"models/{ticker}_lstm_model.keras",
        "scaler": f"models/{ticker}_scaler.pkl",
        "cols": f"models/{ticker}_feature_columns.json",
        "meta": f"models/{ticker}_error_std.json",
    }


def _existing_artifact_paths(ticker: str) -> list[Path]:
    return [Path(path) for path in _artifact_paths(ticker).values() if Path(path).exists()]


def _artifact_status(ticker: str) -> tuple[bool, str]:
    paths = _artifact_paths(ticker)
    missing = [name for name, path in paths.items() if not os.path.exists(path)]
    if missing:
        return True, f"missing artifacts: {', '.join(missing)}"

    artifact_files = _existing_artifact_paths(ticker)
    if not artifact_files:
        return True, "artifacts unavailable"

    oldest_artifact_mtime = min(path.stat().st_mtime for path in artifact_files)

    updated_dependencies = []
    for rel_path in ARTIFACT_DEPENDENCY_FILES:
        dep_path = Path(rel_path)
        # Git checkouts can assign source files slightly newer timestamps than
        # bundled model artifacts even though they belong to the same release.
        if (
            dep_path.exists()
            and dep_path.stat().st_mtime
            > oldest_artifact_mtime + ARTIFACT_MTIME_GRACE_SECONDS
        ):
            updated_dependencies.append(rel_path)

    if updated_dependencies:
        return True, f"pipeline updated: {', '.join(updated_dependencies[:3])}"

    return False, "artifacts current"


def load_artifacts(ticker: str):
    paths = _artifact_paths(ticker)
    missing = [k for k, p in paths.items() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required artifacts for this ticker:\n"
            + "\n".join([f"- {k}: {paths[k]}" for k in missing])
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


def ensure_artifacts_for_ticker(ticker: str, prediction_date: date):
    need_train, reason = _artifact_status(ticker)
    if not need_train:
        return

    with st.spinner(f"Training model for {ticker} ({reason})..."):
        _, metrics = train_stock_model(
            ticker=ticker,
            prediction_date=prediction_date,
            future_days=FUTURE_DAYS,
            n_steps=N_STEPS,
            test_size=0.2,
            epochs=EPOCHS,
            batch_size=32,
            learning_rate=1e-3,
            verbose=0,
        )

    st.success(
        f"Model refreshed for {ticker}. "
        f"Test MAE={metrics['test_mae']:.5f}, "
        f"train/test seq={metrics['num_train_sequences']}/{metrics['num_test_sequences']}"
    )


@st.cache_data(show_spinner=False)
def load_feature_dataframe(
    ticker: str,
    prediction_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds inference feature dataframe:
      - technical indicators
      - daily sentiment features aligned by date
    Returns:
      feats_with_sentiment, scored_headlines
    """
    raw = get_stock_data(ticker, prediction_date)
    feats = add_technical_indicators(raw)

    sent_daily, scored_headlines = build_daily_sentiment_features(
        ticker=ticker,
        market_index=feats.index,
        max_days_back_news_pull=SENTIMENT_PULL_DAYS,
        company_name=None,
    )

    feats = feats.join(sent_daily, how="left")
    feats["Sentiment_MeanCompound"] = feats["Sentiment_MeanCompound"].fillna(0.0)
    feats["Sentiment_HeadlineCount"] = feats["Sentiment_HeadlineCount"].fillna(0.0)
    feats["Sentiment_Vote"] = feats["Sentiment_Vote"].fillna(0.0)

    return feats, scored_headlines


def predict_return(model, scaler, feature_columns, feats_df: pd.DataFrame) -> float:
    """
    Uses last N_STEPS rows to predict FUTURE_DAYS return.
    """
    if len(feats_df) < N_STEPS:
        raise ValueError(
            f"Not enough feature rows ({len(feats_df)}) to build a {N_STEPS}-step sequence."
        )

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


def _coalesce(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _format_currency(value, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"${float(value):,.{decimals}f}"


def _format_large_number(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"

    value = float(value)
    abs_value = abs(value)
    if abs_value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if abs_value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if abs_value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,.0f}"


def _format_volume(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{int(float(value)):,}"


def _format_ratio(value) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.2f}"


def _format_earnings_date(value) -> str:
    if value is None:
        return "N/A"

    if isinstance(value, (list, tuple)) and value:
        value = value[0]

    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return "N/A"
    return ts.strftime("%b %d, %Y")


def _vote_meta(vote: int) -> tuple[str, str]:
    if vote > 0:
        return "Positive / +1", "positive"
    if vote < 0:
        return "Negative / -1", "negative"
    return "Neutral / 0", "neutral"


def _safe_vote_value(value) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


@st.cache_data(show_spinner=False, ttl=1800)
def fetch_stock_overview(ticker: str) -> dict:
    overview = {
        "company_name": None,
        "ticker": ticker,
        "current_price": None,
        "price_change": None,
        "price_change_pct": None,
        "market_cap": None,
        "pe_ratio": None,
        "forward_pe": None,
        "volume": None,
        "week_52_range": None,
        "earnings_date": None,
        "sector": None,
        "industry": None,
        "summary": None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info if hasattr(stock, "info") else {}
        fast_info = getattr(stock, "fast_info", None)
        hist = stock.history(period="5d", interval="1d", auto_adjust=False)

        if hist is not None and not hist.empty:
            current_close = float(hist["Close"].dropna().iloc[-1])
            prev_close = float(hist["Close"].dropna().iloc[-2]) if len(hist["Close"].dropna()) > 1 else None
            overview["current_price"] = current_close

            if prev_close not in (None, 0):
                change = current_close - prev_close
                overview["price_change"] = change
                overview["price_change_pct"] = change / prev_close

        current_price = _coalesce(
            overview["current_price"],
            getattr(fast_info, "last_price", None) if fast_info is not None else None,
            info.get("currentPrice") if isinstance(info, dict) else None,
            info.get("regularMarketPrice") if isinstance(info, dict) else None,
        )
        overview["current_price"] = current_price

        low_52 = _coalesce(
            info.get("fiftyTwoWeekLow") if isinstance(info, dict) else None,
            getattr(fast_info, "year_low", None) if fast_info is not None else None,
        )
        high_52 = _coalesce(
            info.get("fiftyTwoWeekHigh") if isinstance(info, dict) else None,
            getattr(fast_info, "year_high", None) if fast_info is not None else None,
        )
        if low_52 is not None and high_52 is not None:
            overview["week_52_range"] = f"{_format_currency(low_52)} - {_format_currency(high_52)}"

        if isinstance(info, dict):
            overview["company_name"] = _coalesce(
                info.get("longName"),
                info.get("shortName"),
                ticker,
            )
            overview["market_cap"] = _coalesce(info.get("marketCap"))
            overview["pe_ratio"] = _coalesce(info.get("trailingPE"))
            overview["forward_pe"] = _coalesce(info.get("forwardPE"))
            overview["volume"] = _coalesce(
                info.get("volume"),
                info.get("regularMarketVolume"),
            )
            overview["earnings_date"] = _coalesce(
                info.get("earningsDate"),
                info.get("earningsTimestamp"),
            )
            overview["sector"] = _coalesce(info.get("sector"), info.get("category"))
            overview["industry"] = info.get("industry")
            overview["summary"] = _coalesce(
                info.get("longBusinessSummary"),
                info.get("description"),
            )

        if overview["company_name"] is None:
            overview["company_name"] = ticker

    except Exception as exc:
        overview["error"] = str(exc)

    return overview


def render_stock_overview(overview: dict):
    st.subheader("Stock Overview")

    company_name = overview.get("company_name") or overview.get("ticker", "")
    ticker = overview.get("ticker", "")
    current_price = overview.get("current_price")
    price_change = overview.get("price_change")
    price_change_pct = overview.get("price_change_pct")

    headline_left, headline_right = st.columns([2.2, 1.2])
    with headline_left:
        st.markdown(f"### {company_name}")
        meta_parts = [ticker]
        if overview.get("sector"):
            meta_parts.append(str(overview["sector"]))
        if overview.get("industry"):
            meta_parts.append(str(overview["industry"]))
        st.caption(" | ".join(meta_parts))

    with headline_right:
        if current_price is not None and not pd.isna(current_price):
            delta = None
            if price_change is not None and price_change_pct is not None:
                delta = f"{price_change:+.2f} ({price_change_pct:+.2%})"
            st.metric("Current Price", _format_currency(current_price), delta=delta)
        else:
            st.metric("Current Price", "N/A")

    stat_cols = st.columns(4)
    stat_cols[0].metric("Market Cap", _format_large_number(overview.get("market_cap")))
    stat_cols[1].metric("PE Ratio", _format_ratio(overview.get("pe_ratio")))
    stat_cols[2].metric("Forward PE", _format_ratio(overview.get("forward_pe")))
    stat_cols[3].metric("Volume", _format_volume(overview.get("volume")))

    detail_cols = st.columns(2)
    detail_cols[0].metric("52-Week Range", overview.get("week_52_range") or "N/A")
    detail_cols[1].metric("Earnings Date", _format_earnings_date(overview.get("earnings_date")))

    summary = overview.get("summary")
    if summary:
        st.caption(summary)
    elif overview.get("error"):
        st.caption("Company profile details were unavailable for this ticker right now.")
    else:
        st.caption("Company profile details are limited for this ticker.")


def render_sentiment_summary(sent_summary):
    st.subheader("Sentiment Diagnostics")
    s1, s2 = st.columns(2)
    s1.metric("Mean Compound", f"{sent_summary.mean_compound:+.3f}")
    s2.metric("Majority Vote", _vote_meta(sent_summary.majority_vote)[0])

    if sent_summary.headline_count == 0:
        st.info(
            "No headlines were found in the recent sentiment window. "
            "Sentiment features defaulted to neutral values."
        )


def render_headline_cards(headlines_scored: pd.DataFrame, limit: int = 12):
    st.subheader("Recent Headlines")

    if headlines_scored is None or headlines_scored.empty:
        st.caption("No recent headlines were available for this ticker.")
        return

    show_df = headlines_scored.copy().head(limit)
    show_df["published_at"] = pd.to_datetime(
        show_df["published_at"], utc=True, errors="coerce"
    )

    st.markdown(
        """
        <style>
        .headline-card {
            border: 1px solid rgba(49, 51, 63, 0.12);
            border-radius: 14px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.7rem;
            background: linear-gradient(180deg, rgba(250,252,255,0.96), rgba(245,247,250,0.96));
        }
        .headline-meta {
            display: flex;
            justify-content: space-between;
            gap: 0.75rem;
            align-items: center;
            margin-bottom: 0.35rem;
            flex-wrap: wrap;
            font-size: 0.82rem;
            color: #5b6472;
        }
        .headline-source {
            font-weight: 600;
            color: #1f2937;
        }
        .headline-title {
            color: #111827;
            font-size: 0.98rem;
            line-height: 1.4;
            margin: 0;
        }
        .headline-title a {
            color: inherit;
            text-decoration: none;
        }
        .headline-title a:hover {
            text-decoration: underline;
        }
        .vote-pill {
            display: inline-block;
            padding: 0.18rem 0.55rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
        }
        .vote-pill.positive {
            background: rgba(22, 163, 74, 0.12);
            color: #15803d;
        }
        .vote-pill.neutral {
            background: rgba(100, 116, 139, 0.14);
            color: #475569;
        }
        .vote-pill.negative {
            background: rgba(220, 38, 38, 0.12);
            color: #b91c1c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    for _, row in show_df.iterrows():
        published = row.get("published_at")
        published_text = (
            published.strftime("%Y-%m-%d %H:%M UTC")
            if pd.notna(published)
            else "Unknown time"
        )
        source = str(row.get("source") or "Unknown source")
        title = str(row.get("title") or "Untitled headline")
        link = str(row.get("link") or "").strip()
        vote_text, vote_class = _vote_meta(_safe_vote_value(row.get("vote", 0)))

        safe_title = escape(title)
        safe_source = escape(source)
        safe_link = escape(link, quote=True)
        if link:
            title_html = f'<a href="{safe_link}" target="_blank">{safe_title}</a>'
        else:
            title_html = safe_title

        st.markdown(
            f"""
            <div class="headline-card">
                <div class="headline-meta">
                    <div>{published_text} <span class="headline-source">| {safe_source}</span></div>
                    <span class="vote-pill {vote_class}">{vote_text}</span>
                </div>
                <p class="headline-title">{title_html}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ----
# UI
# ----
st.title("Stock Price Forecaster")
st.caption(
    "Forecasts the next 5-business-day move using an LSTM trained on price history, "
    "technical indicators, and daily news sentiment signals."
)
st.caption(
    "Experimental and educational only — model forecasts can be wrong and are not financial advice."
)

st.sidebar.header("Forecast")
ticker = st.sidebar.text_input("Ticker", "AAPL").upper().strip()
today_date = datetime.now().date()
latest_backtest_date = (
    pd.Timestamp(today_date) - pd.tseries.offsets.BDay(FUTURE_DAYS)
).date()

prediction_date = st.sidebar.date_input(
    "Prediction Date",
    today_date,
    max_value=today_date,
)

enable_backtest = st.sidebar.checkbox(
    "Enable backtest",
    value=True,
)

if prediction_date > latest_backtest_date:
    st.sidebar.caption(
        f"Backtest comparison is only available through {latest_backtest_date} "
        "because newer forecasts do not have 5 future trading days of actual data yet."
    )

run_btn = st.sidebar.button("Run Forecast", use_container_width=True)


if run_btn:
    if not ticker:
        st.error("Please enter a ticker.")
        st.stop()

    try:
        ensure_artifacts_for_ticker(
            ticker=ticker,
            prediction_date=prediction_date,
        )

        with st.spinner("Loading trained artifacts..."):
            model, scaler, feature_columns, error_std, ci95 = load_artifacts(ticker)

        with st.spinner("Loading price data and sentiment features..."):
            feats, headlines_scored = load_feature_dataframe(
                ticker=ticker,
                prediction_date=prediction_date,
            )

        with st.spinner("Fetching company overview..."):
            overview = fetch_stock_overview(ticker)

        with st.spinner("Predicting with LSTM..."):
            pred_return = predict_return(model, scaler, feature_columns, feats)

        current_close = float(feats["Close"].iloc[-1])
        pred_price = price_from_return(current_close, pred_return)
        lower_price = price_from_return(current_close, pred_return - ci95)
        upper_price = price_from_return(current_close, pred_return + ci95)

        sent_summary = summarize_sentiment_until_date(
            headlines_scored,
            prediction_date=prediction_date,
            max_days_back=SENTIMENT_LOOKBACK_DAYS,
        )

        render_stock_overview(overview)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Current Close", f"${current_close:,.2f}")
        c2.metric("Predicted 5D Return", f"{pred_return:+.2%}")
        c3.metric("Predicted Price (5D)", f"${pred_price:,.2f}")
        c4.metric("95% Range (Price)", f"${lower_price:,.2f} - ${upper_price:,.2f}")

        render_sentiment_summary(sent_summary)
        render_headline_cards(headlines_scored)

        st.subheader("Price History + Forecast Marker")
        fig, ax = plt.subplots(figsize=(12, 5))
        recent = feats.tail(120)
        ax.plot(recent.index, recent["Close"], label="Close", linewidth=2, color="#0f766e")
        ax.axhline(
            current_close,
            linestyle="--",
            color="#64748b",
            alpha=0.8,
            label="Current Close",
        )
        ax.axhline(
            pred_price,
            linestyle="--",
            color="#15803d",
            alpha=0.9,
            label="Predicted Price (5D)",
        )

        ax.set_title(f"{ticker} Close Price (recent) and 5D Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)

        if enable_backtest and prediction_date <= latest_backtest_date:
            st.subheader("Backtest (single point)")
            future_date = pd.to_datetime(prediction_date) + pd.tseries.offsets.BDay(FUTURE_DAYS)
            try:
                actual_df = get_stock_data(ticker, future_date.date())
                actual_close = float(actual_df["Close"].iloc[-1])

                err_lstm = abs(pred_price - actual_close)

                b1, b2 = st.columns(2)
                b1.metric(f"Actual Close on {future_date.date()}", f"${actual_close:,.2f}")
                b2.metric("Abs Error", f"${err_lstm:,.2f}")
            except Exception as exc:
                st.warning(f"Backtest failed (often due to missing future data): {exc}")
        elif enable_backtest:
            st.subheader("Backtest (single point)")
            st.info(
                f"Forecast generated for {prediction_date}, but a 5-business-day backtest is not available yet. "
                f"Choose a prediction date on or before {latest_backtest_date} to compare against actual data."
            )

    except Exception as exc:
        st.error(str(exc))
        st.stop()
