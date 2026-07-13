"""
sentiment.py — Sentiment analysis pipeline using VADER + label aggregation.

Strategy:
    - Use pre-labeled Kaggle sentiment (positive/neutral/negative) as primary signal.
    - Apply VADER for scoring unlabeled headlines.
    - Aggregate all scores per trading day to produce daily sentiment features.
"""

import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.utils import PROC_DIR, get_logger

logger = get_logger(__name__)

_vader = SentimentIntensityAnalyzer()


# ── VADER scoring ─────────────────────────────────────────────────────────────

def vader_score(text: str) -> dict:
    """
    Return VADER compound, positive, neutral, negative scores for a headline.

    Returns dict with keys: compound, pos, neu, neg
    """
    return _vader.polarity_scores(str(text))


def add_vader_scores(df: pd.DataFrame, text_col: str = "headline_clean") -> pd.DataFrame:
    """
    Append VADER scores to a news DataFrame.

    Parameters
    ----------
    df       : pd.DataFrame  must contain `text_col`
    text_col : str

    Returns
    -------
    pd.DataFrame  with new columns: vader_compound, vader_pos, vader_neu, vader_neg
    """
    logger.info("Computing VADER scores …")
    scores = df[text_col].apply(vader_score).apply(pd.Series)
    scores.columns = ["vader_compound", "vader_pos", "vader_neu", "vader_neg"]
    df = pd.concat([df.reset_index(drop=True), scores], axis=1)
    logger.info("VADER scoring complete.")
    return df


# ── Label → score mapping ─────────────────────────────────────────────────────

LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

def sentiment_label_to_score(label: str) -> float:
    """Convert string label to numeric score."""
    return LABEL_SCORE.get(str(label).strip().lower(), 0.0)


# ── Daily aggregation ─────────────────────────────────────────────────────────

def aggregate_daily_sentiment(
    news_df: pd.DataFrame,
    date_col: str = "Date",
    score_col: str = "sentiment_score",
    label_col: str = "sentiment_label",
) -> pd.DataFrame:
    """
    Aggregate per-headline sentiment into per-day features.

    Input DataFrame must have:
        - `date_col`  : trading date (datetime or parseable string)
        - `score_col` : numeric sentiment score (-1, 0, 1)
        - `label_col` : string label (positive/neutral/negative)

    Returns
    -------
    pd.DataFrame indexed by Date with columns:
        avg_sentiment        — mean score across all headlines that day
        sentiment_volatility — std of scores (0 if only one headline)
        pos_count            — number of positive headlines
        neg_count            — number of negative headlines
        neu_count            — number of neutral headlines
        total_news           — total headlines that day
    """
    logger.info("Aggregating daily sentiment …")

    df = news_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    agg = df.groupby(date_col).agg(
        avg_sentiment        = (score_col, "mean"),
        sentiment_volatility = (score_col, "std"),
        total_news           = (score_col, "count"),
        pos_count            = (label_col, lambda x: (x.str.lower() == "positive").sum()),
        neg_count            = (label_col, lambda x: (x.str.lower() == "negative").sum()),
        neu_count            = (label_col, lambda x: (x.str.lower() == "neutral").sum()),
    ).reset_index()

    # std is NaN when only 1 article → set to 0
    agg["sentiment_volatility"] = agg["sentiment_volatility"].fillna(0.0)

    logger.info(f"Daily sentiment aggregated: {len(agg)} days, "
                f"date range {agg[date_col].min().date()} → {agg[date_col].max().date()}")
    return agg


def build_sentiment_dataset(news_df: pd.DataFrame, save: bool = True) -> pd.DataFrame:
    """
    Full pipeline: score headlines → aggregate to daily → optionally save.

    Parameters
    ----------
    news_df : pd.DataFrame  Output of preprocessor.preprocess_news()
                            Must have: headline_clean, sentiment_label, sentiment_score
                            Optionally: Date (if news has timestamps)

    Returns
    -------
    Daily aggregated sentiment DataFrame.

    Notes
    -----
    The Kaggle dataset has no timestamps — we cannot aggregate by real dates.
    In that case, we return the scored news for the feature-engineering step
    to merge via a date range assigned externally.
    """
    df = add_vader_scores(news_df)

    if "Date" in df.columns:
        daily = aggregate_daily_sentiment(df)
        if save:
            out = PROC_DIR / "daily_sentiment.csv"
            daily.to_csv(out, index=False)
            logger.info(f"Saved → {out}")
        return daily
    else:
        logger.info("No Date column found — returning scored news without date aggregation.")
        if save:
            out = PROC_DIR / "news_scored.csv"
            df.to_csv(out, index=False)
            logger.info(f"Saved → {out}")
        return df


# ── Merge with stock data ─────────────────────────────────────────────────────

def merge_sentiment_with_stock(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    date_col: str = "Date",
) -> pd.DataFrame:
    """
    Left-join stock dataframe with daily sentiment features on Date.

    Missing sentiment days (weekends, no news) are filled with 0
    (neutral sentiment, no headlines).

    Parameters
    ----------
    stock_df     : pd.DataFrame  preprocessed stock data with Date column
    sentiment_df : pd.DataFrame  output of aggregate_daily_sentiment()

    Returns
    -------
    pd.DataFrame  merged dataframe
    """
    stock_df = stock_df.copy()
    sentiment_df = sentiment_df.copy()

    stock_df[date_col]     = pd.to_datetime(stock_df[date_col])
    sentiment_df[date_col] = pd.to_datetime(sentiment_df[date_col])

    merged = stock_df.merge(sentiment_df, on=date_col, how="left")

    # Fill missing sentiment with neutral
    fill_cols = ["avg_sentiment", "sentiment_volatility",
                 "pos_count", "neg_count", "neu_count", "total_news"]
    for col in fill_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    logger.info(f"Merged dataset: {len(merged)} rows, {merged.shape[1]} columns")
    return merged
