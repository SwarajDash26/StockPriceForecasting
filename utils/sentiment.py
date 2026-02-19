from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf

import nltk
from nltk.sentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentSummary:
    headline_count: int
    mean_compound: float
    pos_frac: float
    neg_frac: float
    neu_frac: float


def _get_vader() -> SentimentIntensityAnalyzer:
    """
    Returns a VADER sentiment analyzer.
    If the lexicon isn't installed, raises a helpful error.
    """
    try:
        return SentimentIntensityAnalyzer()
    except LookupError as e:
        raise RuntimeError(
            "VADER lexicon not found. Run:\n"
            "  pip install nltk\n"
            "  python -c \"import nltk; nltk.download('vader_lexicon')\""
        ) from e


def fetch_yfinance_headlines(ticker: str, max_items: int = 50) -> pd.DataFrame:
    """
    Fetch recent news headlines for a ticker using yfinance.
    Returns a DataFrame with columns: published_at (UTC), title, source, link.

    Note: yfinance news is typically *recent* only and may be sparse for some tickers.
    """
    t = yf.Ticker(ticker)
    news = getattr(t, "news", None)
    if news is None:
        return pd.DataFrame(columns=["published_at", "title", "source", "link"])

    rows = []
    for item in news[:max_items]:
        title = item.get("title") or ""
        provider_time = item.get("providerPublishTime", None)  # epoch seconds
        link = item.get("link") or item.get("url") or ""
        source = ""
        publisher = item.get("publisher")
        if isinstance(publisher, str):
            source = publisher
        elif isinstance(publisher, dict):
            source = publisher.get("title", "") or publisher.get("name", "") or ""

        if provider_time is not None:
            published_at = pd.to_datetime(provider_time, unit="s", utc=True)
        else:
            # If missing, skip (can't time-align)
            continue

        rows.append(
            {
                "published_at": published_at,
                "title": title.strip(),
                "source": source,
                "link": link,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["published_at", "title", "source", "link"])

    df = df.dropna(subset=["published_at", "title"])
    df = df[df["title"].astype(str).str.len() > 0]
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def score_headlines_vader(headlines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds VADER sentiment scores to a headlines dataframe.
    Adds columns: compound, pos, neu, neg
    """
    if headlines_df is None or len(headlines_df) == 0:
        return pd.DataFrame(columns=["published_at", "title", "source", "link", "compound", "pos", "neu", "neg"])

    df = headlines_df.copy()
    vader = _get_vader()

    scores = df["title"].astype(str).apply(vader.polarity_scores)
    df["compound"] = scores.apply(lambda s: float(s["compound"]))
    df["pos"] = scores.apply(lambda s: float(s["pos"]))
    df["neu"] = scores.apply(lambda s: float(s["neu"]))
    df["neg"] = scores.apply(lambda s: float(s["neg"]))

    return df


def summarize_sentiment_until_date(
    scored_headlines_df: pd.DataFrame,
    prediction_date,
    max_days_back: int = 7,
) -> SentimentSummary:
    """
    Summarize sentiment using headlines up to and including prediction_date.
    We look back max_days_back calendar days from prediction_date.

    Returns a small set of features that are easy to demo and explain.
    """
    if scored_headlines_df is None or len(scored_headlines_df) == 0:
        return SentimentSummary(0, 0.0, 0.0, 0.0, 0.0)

    df = scored_headlines_df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)

    # Define window: (prediction_date - max_days_back) .. (prediction_date end)
    pred_day = pd.to_datetime(prediction_date)
    window_start = (pred_day - pd.Timedelta(days=max_days_back)).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")
    window_end = (pred_day + pd.Timedelta(days=1)).tz_localize("UTC", nonexistent="shift_forward", ambiguous="NaT")

    # If pred_day was already tz-aware, tz_localize will fail; handle gracefully
    if getattr(pred_day, "tzinfo", None) is not None:
        window_start = pred_day - pd.Timedelta(days=max_days_back)
        window_end = pred_day + pd.Timedelta(days=1)

    df = df[(df["published_at"] >= window_start) & (df["published_at"] < window_end)]
    if len(df) == 0:
        return SentimentSummary(0, 0.0, 0.0, 0.0, 0.0)

    mean_compound = float(df["compound"].mean())

    # Simple fractions using compound sign
    pos_frac = float((df["compound"] > 0.05).mean())
    neg_frac = float((df["compound"] < -0.05).mean())
    neu_frac = float(1.0 - pos_frac - neg_frac)

    return SentimentSummary(
        headline_count=int(len(df)),
        mean_compound=mean_compound,
        pos_frac=pos_frac,
        neg_frac=neg_frac,
        neu_frac=neu_frac,
    )


if __name__ == "__main__":
    import pandas as pd

    ticker = "AAPL"
    prediction_date = pd.to_datetime("today").date()

    h = fetch_yfinance_headlines(ticker, max_items=25)
    hs = score_headlines_vader(h)
    s = summarize_sentiment_until_date(hs, prediction_date, max_days_back=7)

    print(hs.head(5)[["published_at", "title", "compound"]])
    print(s)