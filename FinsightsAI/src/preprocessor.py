"""
preprocessor.py — Clean OHLCV data and financial news; engineer base features.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from src.utils import PROC_DIR, get_logger

logger = get_logger(__name__)

# ── Stock preprocessing ───────────────────────────────────────────────────────

def preprocess_stock(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Clean and prepare raw OHLCV data for feature engineering.

    Steps:
        1. Parse dates, sort chronologically.
        2. Forward-fill then backward-fill missing OHLCV values.
        3. Compute daily return.
        4. Add lag-1, lag-3, lag-5 return features.
        5. Create binary target: 1 if next-day close > today's close, else 0.

    Parameters
    ----------
    df     : pd.DataFrame  Raw OHLCV from data_loader.fetch_stock_data()
    ticker : str           Stock ticker symbol

    Returns
    -------
    pd.DataFrame  Clean dataframe with returns, lags, and target variable.
    """
    logger.info(f"[{ticker}] Preprocessing stock data — {len(df)} rows")

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Handle missing OHLCV
    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    df[ohlcv_cols] = df[ohlcv_cols].ffill().bfill()

    # Daily return
    df["daily_return"] = df["Close"].pct_change()

    # Lag features (using only past data → no leakage)
    df["lag_1"]  = df["daily_return"].shift(1)
    df["lag_3"]  = df["daily_return"].shift(3)
    df["lag_5"]  = df["daily_return"].shift(5)

    # Target: 1 if tomorrow's close > today's close
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop last row (no future close to form a target) and NaN from lags
    df = df.iloc[:-1].dropna(subset=["lag_5", "daily_return"]).reset_index(drop=True)

    logger.info(f"[{ticker}] After preprocessing: {len(df)} rows | "
                f"Target balance: {df['target'].value_counts().to_dict()}")
    return df


def save_processed_stock(df: pd.DataFrame, ticker: str) -> Path:
    """Save preprocessed stock dataframe to data/processed/<ticker>_processed.csv."""
    out = PROC_DIR / f"{ticker}_processed.csv"
    df.to_csv(out, index=False)
    logger.info(f"[{ticker}] Saved → {out}")
    return out


# ── News preprocessing ────────────────────────────────────────────────────────

_NOISE_RE = re.compile(
    r"http\S+|www\.\S+|"       # URLs
    r"[^a-zA-Z0-9\s\.\,\!\?]"  # non-alphanumeric (keep punctuation)
)

def clean_headline(text: str) -> str:
    """Lowercase, strip URLs and special chars from a headline."""
    text = str(text).lower()
    text = _NOISE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_news(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw news dataframe from data_loader.load_news_csv().

    Steps:
        1. Drop exact duplicate (headline, sentiment_label) pairs.
        2. Clean headline text.
        3. Encode sentiment labels to numeric scores.

    Parameters
    ----------
    df : pd.DataFrame  with columns: headline, sentiment_label

    Returns
    -------
    pd.DataFrame with additional columns: headline_clean, sentiment_score
    """
    logger.info(f"Preprocessing {len(df)} news records …")

    df = df.copy()

    # Deduplicate
    before = len(df)
    df = df.drop_duplicates(subset=["headline", "sentiment_label"]).reset_index(drop=True)
    logger.info(f"Removed {before - len(df)} duplicate records.")

    # Clean text
    df["headline_clean"] = df["headline"].apply(clean_headline)

    # Encode labels
    label_map = {"positive": 1, "neutral": 0, "negative": -1}
    df["sentiment_score"] = df["sentiment_label"].map(label_map)

    unmapped = df["sentiment_score"].isna().sum()
    if unmapped:
        logger.warning(f"{unmapped} rows with unmapped sentiment labels → set to 0 (neutral)")
        df["sentiment_score"] = df["sentiment_score"].fillna(0)

    logger.info(f"News preprocessing complete: {len(df)} records retained.")
    return df


def save_processed_news(df: pd.DataFrame) -> Path:
    """Save preprocessed news dataframe to data/processed/news_processed.csv."""
    out = PROC_DIR / "news_processed.csv"
    df.to_csv(out, index=False)
    logger.info(f"Saved → {out}")
    return out
