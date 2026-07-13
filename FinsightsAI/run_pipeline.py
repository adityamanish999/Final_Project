"""
run_pipeline.py — End-to-end pipeline runner.

Run this script once after downloading data to:
  1. Fetch all stock data from Yahoo Finance
  2. Preprocess stocks + news
  3. Run sentiment analysis
  4. Build features
  5. Train all models
  6. Save models + metrics

Usage:
    python run_pipeline.py --news-path data/raw/all-data.csv --ticker AAPL

Or run for all tickers:
    python run_pipeline.py --news-path data/raw/all-data.csv
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from src.utils import TICKERS, PROC_DIR, MODEL_DIR, get_logger
from src.data_loader       import fetch_all_stocks, load_news_csv
from src.preprocessor      import preprocess_stock, save_processed_stock, preprocess_news, save_processed_news
from src.sentiment         import build_sentiment_dataset, merge_sentiment_with_stock, aggregate_daily_sentiment
from src.feature_engineer  import build_features, save_features, ALL_FEATURES
from src.models            import time_split, train_all, evaluate_all, save_all_models, get_best_model

logger = get_logger("pipeline")


def run(news_path: str | None, tickers: list[str]):
    # ── Step 1: Stock data ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Fetching stock data")
    logger.info("=" * 60)
    all_stock_data = fetch_all_stocks(tickers=tickers)

    # ── Step 2: News preprocessing ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Preprocessing financial news")
    logger.info("=" * 60)

    daily_sentiment = None

    if news_path and Path(news_path).exists():
        raw_news    = load_news_csv(news_path)
        clean_news  = preprocess_news(raw_news)
        save_processed_news(clean_news)

        # Score headlines with VADER
        from src.sentiment import add_vader_scores
        clean_news = add_vader_scores(clean_news)

        # If no date column, generate a synthetic daily sentiment
        # by broadcasting labels across dates for demonstration
        if "Date" not in clean_news.columns:
            logger.info("No Date column in news — creating synthetic daily sentiment for demo …")
            # Assign rolling windows of news to each trading date
            # (In production: match by actual publication date)
            daily_sentiment = _synthetic_daily_sentiment(clean_news, all_stock_data)
        else:
            daily_sentiment = aggregate_daily_sentiment(clean_news)
            out = PROC_DIR / "daily_sentiment.csv"
            daily_sentiment.to_csv(out, index=False)
    else:
        logger.warning("No news CSV path provided — sentiment features will be zero-filled.")

    # ── Step 3–5: Per-ticker processing ────────────────────────────────────
    all_metrics = {}

    for ticker in tickers:
        if ticker not in all_stock_data:
            logger.warning(f"Skipping {ticker} — data not fetched.")
            continue

        logger.info("=" * 60)
        logger.info(f"STEPS 3–5: {ticker}")
        logger.info("=" * 60)

        # Preprocess stock
        stock_df = preprocess_stock(all_stock_data[ticker], ticker)
        save_processed_stock(stock_df, ticker)

        # Merge sentiment
        if daily_sentiment is not None:
            merged = merge_sentiment_with_stock(stock_df, daily_sentiment)
        else:
            merged = stock_df.copy()

        # Feature engineering
        feature_df = build_features(merged, ticker=ticker)
        save_features(feature_df, ticker)

        # Train / evaluate
        try:
            X_train, y_train, X_val, y_val, X_test, y_test, dates_test = time_split(feature_df)
            fitted = train_all(X_train, y_train, X_val, y_val)
            metrics_df = evaluate_all(fitted, X_test, y_test)
            save_all_models(fitted, ticker=ticker)

            metrics_out = MODEL_DIR / f"{ticker}_metrics.csv"
            metrics_df.to_csv(metrics_out)
            logger.info(f"[{ticker}] Metrics:\n{metrics_df.to_string()}")
            all_metrics[ticker] = metrics_df

            best_name, _ = get_best_model(metrics_df, fitted)
            logger.info(f"[{ticker}] Best model: {best_name}")
        except Exception as e:
            logger.error(f"[{ticker}] Training failed: {e}")

    logger.info("=" * 60)
    logger.info("Pipeline complete ✓")
    logger.info("=" * 60)
    return all_metrics


def _synthetic_daily_sentiment(news_df: pd.DataFrame, stock_data: dict) -> pd.DataFrame:
    """
    When news has no timestamps, distribute sentiment statistics across
    all trading dates using a rolling bootstrap for demonstration purposes.
    """
    from src.sentiment import aggregate_daily_sentiment

    # Get all trading dates from stock data
    all_dates = set()
    for df in stock_data.values():
        all_dates.update(pd.to_datetime(df["Date"]).dt.normalize().tolist())
    dates = sorted(all_dates)

    n_news = len(news_df)
    rows = []
    for i, date in enumerate(dates):
        # Sample a window of news for this date (rolling)
        start = (i * 5) % n_news
        window = news_df.iloc[start:start + 5] if start + 5 <= n_news else news_df.iloc[-5:]
        rows.append({
            "Date":                 date,
            "avg_sentiment":        window["sentiment_score"].mean(),
            "sentiment_volatility": window["sentiment_score"].std(ddof=0),
            "pos_count":            (window["sentiment_label"] == "positive").sum(),
            "neg_count":            (window["sentiment_label"] == "negative").sum(),
            "neu_count":            (window["sentiment_label"] == "neutral").sum(),
            "total_news":           len(window),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full FinSight AI pipeline.")
    parser.add_argument("--news-path", type=str, default=None,
                        help="Path to the Kaggle financial news CSV (all-data.csv)")
    parser.add_argument("--tickers", nargs="+", default=TICKERS,
                        help="List of stock tickers to process")
    args = parser.parse_args()

    run(news_path=args.news_path, tickers=args.tickers)
