"""
Utility modules for the Churn Prediction pipeline.

Exports:
    evaluate_model: Compute comprehensive classification metrics.
    plot_roc_curve: Multi-model ROC curve visualization.
    plot_confusion_matrix: Styled confusion matrix heatmap.
    compare_models: Tabular comparison of model results.
"""

from .metrics import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
)

__all__ = [
    "evaluate_model",
    "plot_roc_curve",
    "plot_confusion_matrix",
    "compare_models",
]
