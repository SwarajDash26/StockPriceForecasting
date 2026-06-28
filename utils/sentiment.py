from __future__ import annotations

import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf


# Optional ticker -> company hint map to improve query recall
COMPANY_HINTS = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "TSLA": "Tesla",
    "NVDA": "NVIDIA",
    "NFLX": "Netflix",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "INTC": "Intel",
    "AMD": "AMD",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "WMT": "Walmart",
}

# Turn on if you want console diagnostics
DEBUG_SENTIMENT = False


@dataclass
class SentimentSummary:
    headline_count: int
    mean_compound: float
    pos_frac: float
    neg_frac: float
    neu_frac: float
    majority_vote: int  # -1, 0, +1


def _dbg(msg: str):
    if DEBUG_SENTIMENT:
        print(f"[sentiment] {msg}")


class _FallbackLexiconSentiment:
    """
    Lightweight fallback when VADER cannot be loaded.
    """
    POS_WORDS = {
        "beat", "beats", "growth", "surge", "up", "upgrade", "bullish",
        "profit", "profits", "record", "strong", "outperform", "buy",
        "gain", "gains", "rise", "rises", "rally", "expands",
    }
    NEG_WORDS = {
        "miss", "misses", "drop", "drops", "down", "downgrade", "bearish",
        "loss", "losses", "weak", "underperform", "sell", "fall", "falls",
        "lawsuit", "probe", "cuts", "cut", "slump", "decline",
    }

    def polarity_scores(self, text: str) -> Dict[str, float]:
        tokens = re.findall(r"[A-Za-z']+", str(text).lower())
        if not tokens:
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

        pos = sum(1 for t in tokens if t in self.POS_WORDS)
        neg = sum(1 for t in tokens if t in self.NEG_WORDS)
        total = max(len(tokens), 1)

        pos_ratio = pos / total
        neg_ratio = neg / total
        # Keep compound in [-1, 1]
        compound = float(np.clip((pos - neg) / max(pos + neg, 1), -1.0, 1.0))
        neu_ratio = float(max(0.0, 1.0 - pos_ratio - neg_ratio))
        return {
            "compound": compound,
            "pos": float(pos_ratio),
            "neu": neu_ratio,
            "neg": float(neg_ratio),
        }


def _get_sentiment_engine():
    """
    Tries, in order:
    1) NLTK VADER (existing setup)
    2) Auto-download NLTK vader_lexicon then retry
    3) vaderSentiment package
    4) Internal lexical fallback
    """
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer()
    except LookupError:
        try:
            import nltk
            from nltk.sentiment import SentimentIntensityAnalyzer

            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer()
        except Exception as e:
            _dbg(f"NLTK VADER unavailable after download attempt: {e}")
    except Exception as e:
        _dbg(f"NLTK VADER init failed: {e}")

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer

        return VaderSentimentAnalyzer()
    except Exception as e:
        _dbg(f"vaderSentiment package unavailable: {e}")

    _dbg("Using internal fallback lexicon sentiment scorer.")
    return _FallbackLexiconSentiment()


def _empty_news_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["published_at", "title", "source", "link"])


def _clean_news_rows(rows: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return _empty_news_df()

    df = df.dropna(subset=["published_at", "title"])
    df = df[df["title"].astype(str).str.len() > 0]
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    df = df.drop_duplicates(subset=["title", "published_at"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)
    return df


def _parse_dt(value) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        if isinstance(value, (int, float)):
            return pd.to_datetime(value, unit="s", utc=True, errors="coerce")
        if isinstance(value, str):
            value = value.strip()
            # RFC-822 dates from RSS feeds
            if "," in value and " " in value and ":" in value:
                try:
                    dt = parsedate_to_datetime(value)
                    return pd.to_datetime(dt, utc=True, errors="coerce")
                except Exception:
                    pass
        return pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return pd.NaT


def _strict_date_filter(
    df: pd.DataFrame,
    from_date: Optional[str],
    to_date: Optional[str],
) -> pd.DataFrame:
    """
    Enforce strict in-window filter:
      published_at >= from_date 00:00:00 UTC (if provided)
      published_at <  (to_date + 1 day) UTC (if provided)
    """
    if df is None or len(df) == 0:
        return _empty_news_df()

    out = df.copy()
    out["published_at"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    out = out.dropna(subset=["published_at"])

    if from_date:
        from_ts = pd.to_datetime(from_date, utc=True)
        out = out[out["published_at"] >= from_ts]

    if to_date:
        to_exclusive = pd.to_datetime(to_date, utc=True) + pd.Timedelta(days=1)
        out = out[out["published_at"] < to_exclusive]

    out = out.sort_values("published_at", ascending=False).reset_index(drop=True)
    return out


def _newsapi_call(params: dict) -> Tuple[pd.DataFrame, int, dict]:
    """
    Returns: (df, status_code, payload_json_or_empty)
    """
    try:
        r = requests.get("https://newsapi.org/v2/everything", params=params, timeout=12)
        status = r.status_code

        payload = {}
        try:
            payload = r.json()
        except Exception:
            payload = {}

        if status != 200:
            return _empty_news_df(), status, payload

        articles = payload.get("articles", []) or []
        rows = []
        for a in articles:
            title = (a.get("title") or "").strip()
            published_at = a.get("publishedAt")
            src = a.get("source", {}) or {}
            source = src.get("name", "") if isinstance(src, dict) else ""
            link = a.get("url", "") or ""

            if not title or not published_at:
                continue

            rows.append(
                {
                    "published_at": published_at,
                    "title": title,
                    "source": source,
                    "link": link,
                }
            )

        return _clean_news_rows(rows), status, payload

    except Exception as e:
        _dbg(f"NewsAPI exception: {e}")
        return _empty_news_df(), 0, {}


def fetch_newsapi_headlines(
    ticker: str,
    company_name: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    page_size: int = 100,
    language: str = "en",
) -> pd.DataFrame:
    """
    Strict bounded fetch from NewsAPI.
    IMPORTANT: Never returns out-of-window articles.
    """
    api_key = os.getenv("NEWSAPI_KEY", "").strip()
    if not api_key:
        _dbg("NEWSAPI_KEY missing")
        return _empty_news_df()

    t = ticker.upper().strip()
    comp = (company_name or COMPANY_HINTS.get(t, "")).strip()

    query_parts = []
    if comp:
        query_parts.append(f"\"{comp}\"")
    query_parts.extend([f"\"{t}\"", f"\"${t}\""])
    query = " OR ".join(query_parts)

    params = {
        "apiKey": api_key,
        "q": query,
        "language": language,
        "sortBy": "publishedAt",
        "pageSize": min(max(page_size, 1), 100),
    }
    if from_date:
        params["from"] = from_date
    if to_date:
        params["to"] = to_date

    _dbg(f"NewsAPI bounded query: {query} | from={from_date} to={to_date}")

    df, status, payload = _newsapi_call(params)

    # If bounded request failed due to free-tier history restriction (426),
    # do one unbounded request but STRICTLY post-filter to [from, to].
    # This still guarantees no leakage.
    if status == 426:
        _dbg("NewsAPI 426 (date too old for plan). Trying unbounded request + strict local filter.")
        params_unbounded = {
            "apiKey": api_key,
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "pageSize": min(max(page_size, 1), 100),
        }
        df2, status2, payload2 = _newsapi_call(params_unbounded)
        _dbg(f"Unbounded status={status2}, raw_rows={len(df2)}")
        df2 = _strict_date_filter(df2, from_date, to_date)
        _dbg(f"Unbounded post-filter rows={len(df2)}")
        return df2

    # For normal 200, still enforce strict local filter to be safe.
    if status == 200:
        total = payload.get("totalResults", None)
        _dbg(f"NewsAPI status=200 totalResults={total} raw_rows={len(df)}")
        df = _strict_date_filter(df, from_date, to_date)
        _dbg(f"NewsAPI bounded rows after strict filter={len(df)}")
        return df

    _dbg(f"NewsAPI non-200 status={status}, payload={str(payload)[:250]}")
    return _empty_news_df()


def fetch_yfinance_headlines(ticker: str, max_items: int = 80) -> pd.DataFrame:
    """
    Fallback provider.
    """
    try:
        t = yf.Ticker(ticker)
        news = getattr(t, "news", None)
        if not news:
            return _empty_news_df()

        rows = []
        for item in news[:max_items]:
            # Legacy shape:
            # {"title", "providerPublishTime", "publisher", "link"}
            # Newer shape often nests data under "content".
            content = item.get("content", {}) if isinstance(item, dict) else {}
            if not isinstance(content, dict):
                content = {}

            title = (
                (item.get("title") if isinstance(item, dict) else None)
                or content.get("title")
                or ""
            )
            title = str(title).strip()

            ts = None
            if isinstance(item, dict):
                ts = item.get("providerPublishTime")
            ts = ts or content.get("pubDate") or content.get("displayTime")

            publisher = ""
            if isinstance(item, dict):
                publisher = item.get("publisher", "") or ""
            if not publisher:
                provider = content.get("provider", {})
                if isinstance(provider, dict):
                    publisher = (
                        provider.get("displayName")
                        or provider.get("name")
                        or provider.get("title")
                        or ""
                    )

            # url may be at item.link/url or content.canonicalUrl.url
            link = ""
            if isinstance(item, dict):
                link = item.get("link") or item.get("url") or ""
            if not link:
                canonical = content.get("canonicalUrl", {})
                if isinstance(canonical, dict):
                    link = canonical.get("url", "") or ""

            rows.append(
                {
                    "published_at": _parse_dt(ts),
                    "title": title,
                    "source": publisher or "",
                    "link": link,
                }
            )

        return _clean_news_rows(rows)
    except Exception as e:
        _dbg(f"yfinance fallback exception: {e}")
        return _empty_news_df()


def fetch_google_news_rss_headlines(
    ticker: str,
    company_name: Optional[str] = None,
    max_items: int = 100,
) -> pd.DataFrame:
    """
    Keyless fallback source using Google News RSS search.
    """
    t = ticker.upper().strip()
    comp = (company_name or COMPANY_HINTS.get(t, "")).strip()
    query = f"{comp} {t} stock" if comp else f"{t} stock"
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-US&gl=US&ceid=US:en"

    try:
        r = requests.get(url, timeout=12)
        if r.status_code != 200 or not r.text:
            _dbg(f"Google News RSS status={r.status_code}")
            return _empty_news_df()

        root = ET.fromstring(r.text)
        rows = []

        for item in root.findall(".//item")[:max_items]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub = item.findtext("pubDate")
            source = ""
            src_el = item.find("source")
            if src_el is not None and src_el.text:
                source = src_el.text.strip()

            rows.append(
                {
                    "published_at": _parse_dt(pub),
                    "title": title,
                    "source": source,
                    "link": link,
                }
            )

        return _clean_news_rows(rows)
    except Exception as e:
        _dbg(f"Google News RSS exception: {e}")
        return _empty_news_df()


def fetch_headlines(
    ticker: str,
    company_name: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    max_items: int = 100,
) -> pd.DataFrame:
    """
    Source order:
    1) NewsAPI (if NEWSAPI_KEY exists)
    2) yfinance ticker news
    3) Google News RSS
    Result is strict-filtered to [from_date, to_date].
    """
    dfs: List[pd.DataFrame] = []

    df_newsapi = fetch_newsapi_headlines(
        ticker=ticker,
        company_name=company_name,
        from_date=from_date,
        to_date=to_date,
        page_size=max_items,
    )
    if len(df_newsapi) > 0:
        dfs.append(df_newsapi)

    df_yf = fetch_yfinance_headlines(ticker, max_items=min(max_items, 120))
    if len(df_yf) > 0:
        dfs.append(df_yf)

    df_google = fetch_google_news_rss_headlines(
        ticker=ticker,
        company_name=company_name,
        max_items=min(max_items, 120),
    )
    if len(df_google) > 0:
        dfs.append(df_google)

    if len(dfs) == 0:
        _dbg("No headlines from any provider.")
        return _empty_news_df()

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = _clean_news_rows(all_df.to_dict(orient="records"))
    all_df = _strict_date_filter(all_df, from_date, to_date)
    if len(all_df) > max_items:
        all_df = all_df.head(max_items).copy()

    _dbg(f"fetch_headlines final rows={len(all_df)}")
    return all_df


def score_headlines_vader(
    headlines_df: pd.DataFrame,
    threshold: float = 0.05,
) -> pd.DataFrame:
    if headlines_df is None or len(headlines_df) == 0:
        return pd.DataFrame(
            columns=[
                "published_at",
                "title",
                "source",
                "link",
                "compound",
                "pos",
                "neu",
                "neg",
                "vote",
            ]
        )

    vader = _get_sentiment_engine()
    df = headlines_df.copy()

    def _safe_score(text: str) -> Dict[str, float]:
        try:
            s = vader.polarity_scores(str(text))
            return {
                "compound": float(s.get("compound", 0.0)),
                "pos": float(s.get("pos", 0.0)),
                "neu": float(s.get("neu", 1.0)),
                "neg": float(s.get("neg", 0.0)),
            }
        except Exception as e:
            _dbg(f"Scoring exception for title: {e}")
            return {"compound": 0.0, "pos": 0.0, "neu": 1.0, "neg": 0.0}

    scores = df["title"].astype(str).apply(_safe_score)
    df["compound"] = scores.apply(lambda s: s["compound"])
    df["pos"] = scores.apply(lambda s: s["pos"])
    df["neu"] = scores.apply(lambda s: s["neu"])
    df["neg"] = scores.apply(lambda s: s["neg"])

    # discrete headline vote
    df["vote"] = np.where(
        df["compound"] > threshold,
        1,
        np.where(df["compound"] < -threshold, -1, 0),
    ).astype(int)

    return df


def summarize_sentiment_until_date(
    scored_headlines_df: pd.DataFrame,
    prediction_date,
    max_days_back: int = 7,
) -> SentimentSummary:
    """
    Summary window:
      [prediction_date - max_days_back, prediction_date] inclusive by day
    """
    if scored_headlines_df is None or len(scored_headlines_df) == 0:
        return SentimentSummary(0, 0.0, 0.0, 0.0, 0.0, 0)

    df = scored_headlines_df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])

    pred_day = pd.Timestamp(prediction_date)
    pred_day = pred_day.tz_localize("UTC") if pred_day.tzinfo is None else pred_day.tz_convert("UTC")

    start = pred_day - pd.Timedelta(days=max_days_back)
    end_exclusive = pred_day + pd.Timedelta(days=1)

    w = df[(df["published_at"] >= start) & (df["published_at"] < end_exclusive)]

    if len(w) == 0:
        return SentimentSummary(0, 0.0, 0.0, 0.0, 0.0, 0)

    mean_compound = float(w["compound"].mean())
    pos_frac = float((w["vote"] == 1).mean())
    neg_frac = float((w["vote"] == -1).mean())
    neu_frac = float((w["vote"] == 0).mean())

    vote_sum = int(w["vote"].sum())
    majority_vote = 1 if vote_sum > 0 else (-1 if vote_sum < 0 else 0)

    return SentimentSummary(
        headline_count=int(len(w)),
        mean_compound=mean_compound,
        pos_frac=pos_frac,
        neg_frac=neg_frac,
        neu_frac=neu_frac,
        majority_vote=majority_vote,
    )


def build_daily_sentiment_features(
    ticker: str,
    market_index: pd.DatetimeIndex,
    max_days_back_news_pull: int = 45,
    company_name: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build daily sentiment features aligned to market_index dates.

    Pull window:
      from_date = end_date - max_days_back_news_pull
      to_date   = end_date
    """
    if len(market_index) == 0:
        out = pd.DataFrame(index=market_index)
        out["Sentiment_MeanCompound"] = 0.0
        out["Sentiment_HeadlineCount"] = 0.0
        out["Sentiment_Vote"] = 0.0
        return out, pd.DataFrame()

    end_date = pd.to_datetime(market_index.max()).date()
    start_date = (pd.to_datetime(end_date) - pd.Timedelta(days=max_days_back_news_pull)).date()

    _dbg(f"build_daily_sentiment_features ticker={ticker} start={start_date} end={end_date}")

    raw = fetch_headlines(
        ticker=ticker,
        company_name=company_name,
        from_date=str(start_date),
        to_date=str(end_date),
        max_items=100,
    )
    scored = score_headlines_vader(raw)

    daily = pd.DataFrame(index=pd.to_datetime(market_index))
    daily["date"] = daily.index.normalize()

    if len(scored) > 0:
        s = scored.copy()
        s["date"] = pd.to_datetime(s["published_at"], utc=True).dt.tz_convert(None).dt.normalize()

        agg = s.groupby("date").agg(
            Sentiment_MeanCompound=("compound", "mean"),
            Sentiment_HeadlineCount=("vote", "count"),
            vote_sum=("vote", "sum"),
        )
        agg["Sentiment_Vote"] = np.where(
            agg["vote_sum"] > 0, 1,
            np.where(agg["vote_sum"] < 0, -1, 0)
        )
        agg = agg.drop(columns=["vote_sum"])

        daily = daily.merge(agg, left_on="date", right_index=True, how="left")
    else:
        daily["Sentiment_MeanCompound"] = np.nan
        daily["Sentiment_HeadlineCount"] = np.nan
        daily["Sentiment_Vote"] = np.nan

    # neutral defaults on no-news days
    daily["Sentiment_MeanCompound"] = daily["Sentiment_MeanCompound"].fillna(0.0).astype(float)
    daily["Sentiment_HeadlineCount"] = daily["Sentiment_HeadlineCount"].fillna(0.0).astype(float)
    daily["Sentiment_Vote"] = daily["Sentiment_Vote"].fillna(0.0).astype(float)

    daily = daily.drop(columns=["date"])
    daily.index = pd.to_datetime(daily.index)

    return daily, scored
