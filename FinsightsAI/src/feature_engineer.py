"""
feature_engineer.py — Compute technical indicators and temporal features.

All indicators use only past data (shift-aware), ensuring no look-ahead bias.
"""

import pandas as pd
import numpy as np
from src.utils import PROC_DIR, get_logger

logger = get_logger(__name__)


# ── Moving Averages ───────────────────────────────────────────────────────────

def add_sma(df: pd.DataFrame, windows: list[int] = [5, 10, 20]) -> pd.DataFrame:
    """Simple Moving Average over Close price for multiple windows."""
    for w in windows:
        df[f"sma_{w}"] = df["Close"].rolling(window=w, min_periods=w).mean()
    return df


def add_ema(df: pd.DataFrame, spans: list[int] = [12, 26]) -> pd.DataFrame:
    """Exponential Moving Average over Close price."""
    for s in spans:
        df[f"ema_{s}"] = df["Close"].ewm(span=s, adjust=False).mean()
    return df


# ── RSI ───────────────────────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Relative Strength Index (Wilder's smoothing method).
    RSI ranges 0–100. Values > 70 = overbought, < 30 = oversold.
    """
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    return df


# ── MACD ──────────────────────────────────────────────────────────────────────

def add_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD = EMA(fast) − EMA(slow)
    Signal line = EMA(MACD, signal)
    Histogram = MACD − Signal
    """
    ema_fast   = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow   = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"]    = df["macd"].ewm(span=signal, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]
    return df


# ── Bollinger Bands ───────────────────────────────────────────────────────────

def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """
    Bollinger Bands:
        Middle Band = SMA(20)
        Upper Band  = Middle + 2σ
        Lower Band  = Middle − 2σ
        %B          = (Close − Lower) / (Upper − Lower)
    """
    mid = df["Close"].rolling(window=window, min_periods=window).mean()
    std = df["Close"].rolling(window=window, min_periods=window).std()

    df["bb_upper"]  = mid + num_std * std
    df["bb_lower"]  = mid - num_std * std
    df["bb_middle"] = mid
    df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / mid.replace(0, np.nan)
    df["bb_pct_b"]  = (df["Close"] - df["bb_lower"]) / (
        (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    )
    return df


# ── Rolling Volatility ────────────────────────────────────────────────────────

def add_rolling_volatility(
    df: pd.DataFrame,
    windows: list[int] = [5, 10],
) -> pd.DataFrame:
    """Rolling standard deviation of daily returns."""
    for w in windows:
        df[f"volatility_{w}"] = df["daily_return"].rolling(window=w, min_periods=w).std()
    return df


# ── Temporal Features ─────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """Day of week (0=Mon), month, is_month_end flag."""
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"]  = dt.dt.dayofweek
    df["month"]        = dt.dt.month
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    return df


# ── Full pipeline ─────────────────────────────────────────────────────────────

def build_features(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Apply all technical indicators and temporal features to a preprocessed
    stock DataFrame (with optional sentiment columns already merged).

    Parameters
    ----------
    df     : pd.DataFrame  Output of preprocessor + optional sentiment merge.
             Must contain: Date, Open, High, Low, Close, Volume,
                           daily_return, lag_1, lag_3, lag_5, target.
    ticker : str           For logging only.

    Returns
    -------
    pd.DataFrame  with all engineered features, NaN rows from rolling windows dropped.
    """
    logger.info(f"[{ticker}] Building features …")
    df = df.copy()

    df = add_sma(df)
    df = add_ema(df)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_rolling_volatility(df)
    df = add_temporal_features(df)

    before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info(f"[{ticker}] Features built: {len(df)} rows "
                f"(dropped {before - len(df)} NaN rows from rolling windows)")
    return df


def save_features(df: pd.DataFrame, ticker: str) -> None:
    """Save feature-engineered dataframe to data/processed/<ticker>_features.csv."""
    out = PROC_DIR / f"{ticker}_features.csv"
    df.to_csv(out, index=False)
    logger.info(f"[{ticker}] Saved → {out}")


# ── Feature column registry ───────────────────────────────────────────────────

TECHNICAL_FEATURES = [
    "sma_5", "sma_10", "sma_20",
    "ema_12", "ema_26",
    "rsi",
    "macd", "macd_signal", "macd_histogram",
    "bb_upper", "bb_lower", "bb_middle", "bb_width", "bb_pct_b",
    "volatility_5", "volatility_10",
]

SENTIMENT_FEATURES = [
    "avg_sentiment", "sentiment_volatility",
    "pos_count", "neg_count", "neu_count", "total_news",
]

TEMPORAL_FEATURES = [
    "day_of_week", "month", "is_month_end",
    "lag_1", "lag_3", "lag_5",
]

ALL_FEATURES = TECHNICAL_FEATURES + SENTIMENT_FEATURES + TEMPORAL_FEATURES
