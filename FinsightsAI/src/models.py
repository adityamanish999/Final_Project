"""
models.py — Train, evaluate, and persist machine learning models.

Implements chronological train/val/test splitting to prevent data leakage.
Models: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model  import LogisticRegression
from sklearn.ensemble      import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics       import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.pipeline      import Pipeline
import xgboost  as xgb
import lightgbm as lgb

from src.utils import MODEL_DIR, TRAIN_END, VAL_END, get_logger
from src.feature_engineer import ALL_FEATURES

logger = get_logger(__name__)


# ── Chronological split ───────────────────────────────────────────────────────

def time_split(
    df: pd.DataFrame,
    date_col: str    = "Date",
    train_end: str   = TRAIN_END,
    val_end: str     = VAL_END,
    feature_cols: list[str] = ALL_FEATURES,
    target_col: str  = "target",
):
    """
    Split DataFrame into (X_train, y_train), (X_val, y_val), (X_test, y_test)
    using strict chronological boundaries — NO random shuffling.

    Parameters
    ----------
    df           : pd.DataFrame  Feature-engineered dataframe.
    date_col     : str
    train_end    : str  "YYYY-MM-DD" — last date in training set
    val_end      : str  "YYYY-MM-DD" — last date in validation set
    feature_cols : list[str]
    target_col   : str

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test, dates_test)
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Filter to only available feature columns
    available_features = [c for c in feature_cols if c in df.columns]
    missing = set(feature_cols) - set(available_features)
    if missing:
        logger.warning(f"Missing feature columns (will be ignored): {missing}")

    train_mask = df[date_col] <= train_end
    val_mask   = (df[date_col] > train_end) & (df[date_col] <= val_end)
    test_mask  = df[date_col] > val_end

    X_train = df.loc[train_mask, available_features]
    y_train = df.loc[train_mask, target_col]
    X_val   = df.loc[val_mask,   available_features]
    y_val   = df.loc[val_mask,   target_col]
    X_test  = df.loc[test_mask,  available_features]
    y_test  = df.loc[test_mask,  target_col]
    dates_test = df.loc[test_mask, date_col]

    logger.info(f"Split sizes → Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, dates_test


# ── Model definitions ─────────────────────────────────────────────────────────

def _build_models() -> dict:
    """Return dict of model_name → sklearn-compatible estimator."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=1, random_state=42, verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, verbose=-1,
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────

def train_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val:   pd.DataFrame,
    y_val:   pd.Series,
) -> dict:
    """
    Train all four models and return fitted estimators.

    XGBoost uses the validation set for early stopping.

    Returns
    -------
    dict: model_name → fitted estimator
    """
    models = _build_models()
    fitted = {}

    for name, model in models.items():
        logger.info(f"Training {name} …")
        if name == "XGBoost":
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)

        fitted[name] = model
        logger.info(f"{name} trained ✓")

    return fitted


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "",
) -> dict:
    """
    Compute classification metrics for a fitted model on dataset (X, y).

    Returns
    -------
    dict with: accuracy, precision, recall, f1, roc_auc, confusion_matrix
    """
    y_pred  = model.predict(X)
    y_proba = (
        model.predict_proba(X)[:, 1]
        if hasattr(model, "predict_proba")
        else None
    )

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y, y_pred),              4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y, y_pred, zero_division=0),    4),
        "f1":        round(f1_score(y, y_pred, zero_division=0),        4),
        "roc_auc":   round(roc_auc_score(y, y_proba) if y_proba is not None else float("nan"), 4),
        "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
    }

    logger.info(
        f"[{model_name}] Acc={metrics['accuracy']} | P={metrics['precision']} | "
        f"R={metrics['recall']} | F1={metrics['f1']} | AUC={metrics['roc_auc']}"
    )
    return metrics


def evaluate_all(
    fitted_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all fitted models on the test set.

    Returns
    -------
    pd.DataFrame of metrics, one row per model.
    """
    results = []
    for name, model in fitted_models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        results.append(m)
    df = pd.DataFrame(results).set_index("model")
    return df


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, name: str, ticker: str = "") -> Path:
    """Serialize a fitted model to data/models/<ticker>_<name>.pkl."""
    safe_name = name.replace(" ", "_").lower()
    fname     = f"{ticker}_{safe_name}.pkl" if ticker else f"{safe_name}.pkl"
    out       = MODEL_DIR / fname
    joblib.dump(model, out)
    logger.info(f"Saved {name} → {out}")
    return out


def load_model(name: str, ticker: str = ""):
    """Load a model from data/models/."""
    safe_name = name.replace(" ", "_").lower()
    fname     = f"{ticker}_{safe_name}.pkl" if ticker else f"{safe_name}.pkl"
    path      = MODEL_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def save_all_models(fitted_models: dict, ticker: str = "") -> None:
    """Save all models in the fitted_models dict."""
    for name, model in fitted_models.items():
        save_model(model, name, ticker=ticker)


def get_best_model(metrics_df: pd.DataFrame, fitted_models: dict, metric: str = "f1"):
    """Return the (name, model) pair with the highest value on `metric`."""
    best_name = metrics_df[metric].idxmax()
    logger.info(f"Best model by {metric}: {best_name} ({metrics_df.loc[best_name, metric]})")
    return best_name, fitted_models[best_name]
