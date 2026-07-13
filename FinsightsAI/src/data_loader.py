"""
data_loader.py — Fetch and persist historical OHLCV data from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from src.utils import RAW_DIR, TICKERS, START_DATE, END_DATE, get_logger

logger = get_logger(__name__)


def fetch_stock_data(
    ticker: str,
    start: str = START_DATE,
    end: str = END_DATE,
    save: bool = True,
) -> pd.DataFrame:
    """
    Download daily OHLCV data for a single ticker from Yahoo Finance.

    Parameters
    ----------
    ticker : str   e.g. "AAPL"
    start  : str   ISO date string "YYYY-MM-DD"
    end    : str   ISO date string "YYYY-MM-DD"
    save   : bool  If True, write CSV to data/raw/<ticker>.csv

    Returns
    -------
    pd.DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    logger.info(f"Fetching {ticker} from {start} to {end} …")
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if df.empty:
        logger.warning(f"No data returned for {ticker}.")
        return df

    # Flatten MultiIndex columns (yfinance ≥ 0.2.x sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df.index.name = "Date"
    df = df.reset_index()
    df["Ticker"] = ticker

    if save:
        out_path = RAW_DIR / f"{ticker}_raw.csv"
        df.to_csv(out_path, index=False)
        logger.info(f"Saved → {out_path}  ({len(df)} rows)")

    return df


def fetch_all_stocks(
    tickers: list = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE,
) -> dict[str, pd.DataFrame]:
    """
    Download OHLCV data for every ticker in the list.

    Returns
    -------
    dict mapping ticker → DataFrame
    """
    data = {}
    for ticker in tickers:
        df = fetch_stock_data(ticker, start=start, end=end)
        if not df.empty:
            data[ticker] = df
    logger.info(f"Fetched {len(data)}/{len(tickers)} tickers successfully.")
    return data


def load_news_csv(filepath: str | Path) -> pd.DataFrame:
    """
    Load the Kaggle financial news sentiment CSV.

    Expected columns: sentence (headline text), sentiment (positive/neutral/negative)
    Accepts both comma-separated and @-separated (Financial PhraseBank) formats.

    Returns
    -------
    pd.DataFrame with columns: headline, sentiment_label
    """
    filepath = Path(filepath)
    logger.info(f"Loading news CSV from {filepath} …")

    # Try common delimiters
    for sep in [",", "@", "\t", ";"]:
        try:
            df = pd.read_csv(filepath, sep=sep, header=None, encoding="latin-1")
            if df.shape[1] >= 2:
                break
        except Exception:
            continue

    # Standardize column names
    if df.shape[1] == 2:
        df.columns = ["sentiment_label", "headline"]
    elif df.shape[1] >= 3:
        # Some versions have an extra index col
        df = df.iloc[:, -2:]
        df.columns = ["sentiment_label", "headline"]

    df["headline"] = df["headline"].astype(str).str.strip()
    df["sentiment_label"] = df["sentiment_label"].astype(str).str.strip().str.lower()

    logger.info(f"Loaded {len(df)} news records. Label counts:\n{df['sentiment_label'].value_counts().to_string()}")
    return df


if __name__ == "__main__":
    # Quick test — downloads all tickers and prints head
    all_data = fetch_all_stocks()
    for ticker, df in all_data.items():
        print(f"\n{ticker}:\n{df.head(3)}")
