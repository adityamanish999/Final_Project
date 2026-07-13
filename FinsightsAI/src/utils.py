"""
utils.py — Shared helpers for the financial prediction pipeline.
"""

import os
import logging
from pathlib import Path

# ── Project root paths ────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
PROC_DIR   = DATA_DIR / "processed"
MODEL_DIR  = DATA_DIR / "models"

for _d in [RAW_DIR, PROC_DIR, MODEL_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Stocks & date range ───────────────────────────────────────────────────────
TICKERS    = ["AAPL", "MSFT", "NVDA", "TSLA"]
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"

# ── Train / Val / Test boundaries ────────────────────────────────────────────
TRAIN_END  = "2021-12-31"
VAL_END    = "2022-12-31"
# TEST       = everything after VAL_END

# ── Logger factory ────────────────────────────────────────────────────────────
def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a consistent format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
