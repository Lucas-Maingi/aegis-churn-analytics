"""
Evaluation utilities for the Churn Prediction pipeline.

Provides functions for computing classification metrics, finding optimal
decision thresholds, plotting ROC curves / confusion matrices, and
creating side-by-side model comparison tables.

Typical usage::

    from churn_prediction.utils.metrics import evaluate_model, compare_models

    results = evaluate_model(model, X_test, y_test, model_name="XGBoost")
    comparison_df = compare_models({"XGBoost": results, "LR": lr_results})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Professional plot style
# ---------------------------------------------------------------------------
_PLOT_STYLE = "seaborn-v0_8-darkgrid"

try:
    plt.style.use(_PLOT_STYLE)
except OSError:
    # Fall back gracefully when the exact style sheet is unavailable.
    logger.warning(
        "Plot style '%s' not found – falling back to 'dark_background'.",
        _PLOT_STYLE,
    )
    plt.style.use("dark_background")


# =========================================================================
# Core evaluation
# =========================================================================

def evaluate_model(
    model: Any,
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    model_name: str = "model",
) -> Dict[str, Any]:
    """Compute a comprehensive set of binary-classification metrics.

    Parameters
    ----------
    model:
        A fitted scikit-learn–compatible estimator that exposes
        ``predict`` and ``predict_proba``.
    X_test:
        Test feature matrix.
    y_test:
        Ground-truth binary labels (0 / 1).
    model_name:
        Human-readable identifier used in log messages.

    Returns
    -------
    dict
        Keys: ``model_name``, ``auc_roc``, ``precision``, ``recall``,
        ``f1``, ``accuracy``, ``optimal_threshold``, ``confusion_matrix``,
        ``classification_report``.
    """
    logger.info("Evaluating model: %s", model_name)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- scalar metrics (default threshold 0.5) --------------------------
    auc_roc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    # --- optimal threshold via Youden's J statistic ----------------------
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    optimal_threshold: float = float(thresholds[best_idx])

    # --- confusion matrix & classification report ------------------------
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    logger.info(
        "%s - AUC-ROC: %.4f | F1: %.4f | Precision: %.4f | "
        "Recall: %.4f | Accuracy: %.4f | Optimal threshold: %.4f",
        model_name, auc_roc, f1, precision, recall, accuracy,
        optimal_threshold,
    )

    return {
        "model_name": model_name,
        "auc_roc": auc_roc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "optimal_threshold": optimal_threshold,
        "confusion_matrix": cm,
        "classification_report": report,
    }


# =========================================================================
# ROC curve — multiple models
# =========================================================================

def plot_roc_curve(
    models_dict: Dict[str, Any],
    X_test: np.ndarray | pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot overlaid ROC curves for one or more models.

    Parameters
    ----------
    models_dict:
        ``{name: fitted_model}`` mapping.
    X_test, y_test:
        Held-out test data.
    figsize:
        Matplotlib figure size.
    save_path:
        If provided, the figure is saved to this path (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    color_cycle = plt.cm.Set2(np.linspace(0, 1, max(len(models_dict), 3)))

    for idx, (name, model) in enumerate(models_dict.items()):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(
            fpr, tpr,
            color=color_cycle[idx],
            linewidth=2.5,
            label=f"{name} (AUC = {roc_auc:.4f})",
        )

    # Reference diagonal
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--", color="grey", linewidth=1, alpha=0.7,
        label="Random (AUC = 0.5000)",
    )

    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("ROC Curve Comparison", fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("ROC curve saved to %s", save_path)

    return fig


# =========================================================================
# Confusion matrix heatmap
# =========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    model_name: str = "Model",
    figsize: tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Create a styled confusion-matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    model_name:
        Title prefix.
    figsize:
        Figure size.
    save_path:
        Optional save path (PNG).

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Not Churned", "Churned"]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Tick labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)

    # Annotate cells with counts and percentages
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = count / total * 100
            colour = "white" if count > cm.max() / 2 else "black"
            ax.text(
                j, i,
                f"{count}\n({pct:.1f}%)",
                ha="center", va="center",
                fontsize=13, fontweight="bold",
                color=colour,
            )

    ax.set_xlabel("Predicted Label", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    ax.set_title(
        f"{model_name} — Confusion Matrix",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    return fig


# =========================================================================
# Model comparison table
# =========================================================================

def compare_models(
    results_dict: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """Build a comparison DataFrame from multiple ``evaluate_model`` outputs.

    Parameters
    ----------
    results_dict:
        ``{model_name: evaluate_model_result}`` mapping.

    Returns
    -------
    pd.DataFrame
        Sorted by AUC-ROC descending, with columns for every scalar metric
        and the optimal threshold.
    """
    _METRIC_COLS: List[str] = [
        "auc_roc",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "optimal_threshold",
    ]

    rows: List[Dict[str, Any]] = []
    for name, metrics in results_dict.items():
        row: Dict[str, Any] = {"model_name": name}
        for col in _METRIC_COLS:
            row[col] = metrics.get(col)
        rows.append(row)

    df = (
        pd.DataFrame(rows)
        .set_index("model_name")
        .sort_values("auc_roc", ascending=False)
    )

    logger.info("Model comparison table:\n%s", df.to_string())
    return df
