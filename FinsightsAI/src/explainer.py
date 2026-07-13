"""
explainer.py — SHAP-based model explainability.

Generates global (dataset-level) and local (single-prediction) SHAP explanations
for all four model types: Logistic Regression, Random Forest, XGBoost, LightGBM.
"""

import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.pipeline import Pipeline

from src.utils import MODEL_DIR, get_logger

logger = get_logger(__name__)

_PLOT_DIR = MODEL_DIR / "shap_plots"
_PLOT_DIR.mkdir(parents=True, exist_ok=True)


# ── Explainer factory ─────────────────────────────────────────────────────────

def get_explainer(model, X_background: pd.DataFrame):
    """
    Return the correct SHAP explainer for a given model type.

    - Tree-based (RF, XGBoost, LightGBM) → TreeExplainer
    - Pipeline with LinearModel              → LinearExplainer on the inner clf
    - Fallback                               → KernelExplainer (slow, avoid for large X)
    """
    # Unwrap sklearn Pipeline to get the inner classifier
    inner = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    inner_name = type(inner).__name__

    if inner_name in ("RandomForestClassifier", "XGBClassifier", "LGBMClassifier",
                      "DecisionTreeClassifier", "GradientBoostingClassifier"):
        logger.info(f"Using TreeExplainer for {inner_name}")
        explainer = shap.TreeExplainer(inner)

    elif inner_name in ("LogisticRegression", "LinearSVC", "Ridge"):
        logger.info(f"Using LinearExplainer for {inner_name}")
        # Extract the transformed X for linear models inside a Pipeline
        if isinstance(model, Pipeline):
            X_transformed = model[:-1].transform(X_background)
            masker = shap.maskers.Independent(X_transformed, max_samples=100)
            explainer = shap.LinearExplainer(inner, masker=masker)
        else:
            explainer = shap.LinearExplainer(inner, X_background)

    else:
        logger.warning(f"Unknown model type {inner_name} — using KernelExplainer (slow)")
        predict_fn = lambda x: model.predict_proba(x)[:, 1]
        explainer  = shap.KernelExplainer(predict_fn, shap.sample(X_background, 50))

    return explainer


def compute_shap_values(
    model,
    X: pd.DataFrame,
    X_background: pd.DataFrame | None = None,
) -> shap.Explanation | np.ndarray:
    """
    Compute SHAP values for X.

    Parameters
    ----------
    model        : fitted sklearn-compatible model
    X            : pd.DataFrame  samples to explain
    X_background : pd.DataFrame  background/training data (needed for some explainers)

    Returns
    -------
    shap.Explanation or np.ndarray of SHAP values
    """
    if X_background is None:
        X_background = X

    explainer = get_explainer(model, X_background)

    inner = model.named_steps["clf"] if isinstance(model, Pipeline) else model
    inner_name = type(inner).__name__

    if inner_name in ("LogisticRegression",) and isinstance(model, Pipeline):
        X_input = model[:-1].transform(X)
    else:
        X_input = X

    shap_values = explainer(X_input)
    return shap_values


# ── Global plots ──────────────────────────────────────────────────────────────

def plot_global_importance(
    shap_values,
    feature_names: list[str],
    model_name: str = "model",
    max_display: int = 15,
    save: bool = True,
) -> plt.Figure:
    """
    Bar chart of mean |SHAP| values — global feature importance.

    Returns matplotlib Figure object (for Streamlit rendering).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Handle multi-output SHAP (binary → take class-1 values)
    sv = shap_values
    if hasattr(sv, "values"):
        vals = sv.values
        if vals.ndim == 3:          # (samples, features, classes)
            vals = vals[:, :, 1]
    else:
        vals = np.array(sv)
        if vals.ndim == 3:
            vals = vals[1]

    mean_abs = np.abs(vals).mean(axis=0)
    indices  = np.argsort(mean_abs)[::-1][:max_display]

    ax.barh(
        [feature_names[i] for i in indices[::-1]],
        mean_abs[indices[::-1]],
        color="#4f8ef7",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance — {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save:
        path = _PLOT_DIR / f"{model_name.replace(' ', '_').lower()}_global.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved global plot → {path}")

    return fig


def plot_beeswarm(
    shap_values,
    feature_names: list[str],
    model_name: str = "model",
    max_display: int = 15,
    save: bool = True,
) -> plt.Figure:
    """SHAP beeswarm plot showing direction of feature impact."""
    fig, ax = plt.subplots(figsize=(10, 7))

    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = shap.Explanation(
            values=sv.values[:, :, 1],
            base_values=sv.base_values[:, 1] if sv.base_values.ndim == 2 else sv.base_values,
            data=sv.data,
            feature_names=feature_names,
        )

    shap.plots.beeswarm(sv, max_display=max_display, show=False)

    if save:
        path = _PLOT_DIR / f"{model_name.replace(' ', '_').lower()}_beeswarm.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved beeswarm plot → {path}")

    fig = plt.gcf()
    plt.close()
    return fig


# ── Local (single prediction) plots ──────────────────────────────────────────

def plot_waterfall(
    shap_values,
    sample_idx: int,
    feature_names: list[str],
    model_name: str = "model",
    save: bool = True,
) -> plt.Figure:
    """
    SHAP waterfall plot for a single prediction (sample_idx).

    Shows how each feature contributed to moving the prediction
    away from the base (expected) value.
    """
    sv = shap_values
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv = shap.Explanation(
            values=sv.values[sample_idx, :, 1],
            base_values=sv.base_values[sample_idx, 1] if sv.base_values.ndim == 2
                        else sv.base_values[sample_idx],
            data=sv.data[sample_idx] if sv.data is not None else None,
            feature_names=feature_names,
        )
    else:
        sv = sv[sample_idx]

    shap.plots.waterfall(sv, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Local Explanation — {model_name} | Sample {sample_idx}", y=1.01)

    if save:
        path = _PLOT_DIR / f"{model_name.replace(' ', '_').lower()}_waterfall_{sample_idx}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved waterfall → {path}")

    plt.close()
    return fig


# ── Convenience: run full SHAP analysis for all models ───────────────────────

def run_full_explanation(
    fitted_models: dict,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    feature_names: list[str],
) -> dict:
    """
    Compute and save SHAP global importance plots for every model.

    Returns
    -------
    dict: model_name → {"shap_values": ..., "global_fig": ..., "beeswarm_fig": ...}
    """
    results = {}
    for name, model in fitted_models.items():
        logger.info(f"Running SHAP for {name} …")
        try:
            sv = compute_shap_values(model, X_test, X_background=X_train)
            g_fig = plot_global_importance(sv, feature_names, model_name=name)
            results[name] = {
                "shap_values":   sv,
                "global_fig":    g_fig,
            }
        except Exception as e:
            logger.error(f"SHAP failed for {name}: {e}")

    return results
