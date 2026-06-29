"""
SHAP-based model explainability for the Churn Prediction pipeline.

Provides both **global** feature-importance analysis and **per-prediction**
plain-English explanations that can be surfaced directly by the API.

Typical usage::

    from churn_prediction.models.explainer import ChurnExplainer

    explainer = ChurnExplainer(model, feature_names)
    shap_values = explainer.compute_shap_values(X_test)
    top_global  = explainer.get_top_features(shap_values, feature_names, n=5)
    single_exp  = explainer.explain_single_prediction(X_test[0:1], feature_names)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


# =========================================================================
# Plain-English explanation templates
# =========================================================================

FEATURE_EXPLANATIONS: Dict[str, str] = {
    # Contract features
    "Contract_Month-to-month": (
        "This user is on a month-to-month contract, the strongest churn "
        "indicator. Users on long-term contracts churn 13x less."
    ),
    "Contract_One year": (
        "This user is on a one-year contract, which moderately reduces "
        "churn risk compared to month-to-month plans."
    ),
    "Contract_Two year": (
        "This user has a two-year contract — the most protective factor "
        "against churn."
    ),

    # Tenure & charges
    "tenure": (
        "This user has been a customer for {value} months. "
        "Shorter tenure correlates with higher churn risk."
    ),
    "MonthlyCharges": (
        "Monthly charges of ${value:.2f} are {comparison} the average. "
        "Higher charges increase churn risk."
    ),
    "TotalCharges": (
        "Total lifetime charges of ${value:.2f}. Lower totals often "
        "signal newer, higher-risk customers."
    ),

    # Internet & add-on services
    "InternetService_Fiber optic": (
        "Fiber optic service users churn more frequently, likely due to "
        "higher costs and competitive alternatives."
    ),
    "InternetService_DSL": (
        "DSL internet users tend to churn at lower rates than fiber optic "
        "subscribers."
    ),
    "TechSupport_No": (
        "This user lacks tech support, a key protective service against "
        "churn."
    ),
    "OnlineSecurity_No": (
        "No online security subscription — users without security "
        "services churn at higher rates."
    ),
    "OnlineBackup_No": (
        "No online backup subscription — lacking add-on services "
        "correlates with higher churn."
    ),
    "DeviceProtection_No": (
        "No device protection plan — users without bundled protections "
        "tend to churn more."
    ),
    "StreamingTV_No": (
        "No streaming TV subscription. Fewer services can mean lower "
        "switching cost and higher churn risk."
    ),
    "StreamingMovies_No": (
        "No streaming movies subscription. Engagement with fewer "
        "services weakens retention."
    ),

    # Payment & billing
    "PaymentMethod_Electronic check": (
        "Electronic check payment is associated with higher churn — "
        "possibly indicating less commitment than automatic payments."
    ),
    "PaperlessBilling": (
        "Paperless billing users churn more, possibly due to less "
        "engagement with service communications."
    ),

    # Demographics
    "SeniorCitizen": (
        "Senior citizens churn at a slightly higher rate, potentially "
        "due to price sensitivity or different service needs."
    ),
    "Partner": (
        "Having a partner is mildly protective — shared accounts may "
        "increase retention."
    ),
    "Dependents": (
        "Customers with dependents tend to churn less, likely due to "
        "household service consolidation."
    ),

    # Phone service
    "PhoneService": (
        "Phone service alone has a minor effect, but its interaction "
        "with other features matters."
    ),
    "MultipleLines_No": (
        "Single-line users may have simpler needs and lower switching "
        "costs."
    ),
    "MultipleLines_Yes": (
        "Multiple-line users are somewhat more retained, likely due to "
        "household bundling."
    ),

    # Engineered features
    "tenure_bucket": (
        "Tenure bucket '{value}' — newer customers (0–12 months) are at "
        "significantly higher churn risk."
    ),
    "contract_charge_interaction": (
        "Contract-charge interaction value of {value:.2f} captures the "
        "combined effect of contract type and monthly charges."
    ),
    "avg_monthly_charge": (
        "Average monthly charge of ${value:.2f} — higher averages may "
        "indicate customers who never received discounts."
    ),
    "num_services": (
        "This customer subscribes to {value} services. More services "
        "equals higher engagement and lower churn."
    ),
    "has_protection_bundle": (
        "Protection bundle status: {value}. Customers with the full "
        "bundle (security + support + device protection) churn far less."
    ),
}


# =========================================================================
# ChurnExplainer
# =========================================================================

class ChurnExplainer:
    """SHAP-powered explainability engine for churn models.

    Automatically selects ``TreeExplainer`` for tree-based models
    (XGBoost, LightGBM, RandomForest, …) and ``LinearExplainer`` for
    linear models (Logistic Regression, SGD, …).

    Parameters
    ----------
    model:
        A fitted scikit-learn–compatible estimator.
    feature_names:
        Ordered list of feature column names matching the model's
        input dimensionality.
    """

    _TREE_MODEL_TYPES = (
        "XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
        "GradientBoostingClassifier", "DecisionTreeClassifier",
        "ExtraTreesClassifier",
    )

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
    ) -> None:
        self.model = model
        self.feature_names = feature_names
        self._explainer = self._build_explainer()

    # ------------------------------------------------------------------
    # Explainer factory
    # ------------------------------------------------------------------
    def _build_explainer(self) -> shap.Explainer:
        """Select the appropriate SHAP explainer for the model type."""
        model_type = type(self.model).__name__

        if model_type in self._TREE_MODEL_TYPES:
            logger.info(
                "Using TreeExplainer for model type '%s'.", model_type,
            )
            return shap.TreeExplainer(self.model)

        if isinstance(self.model, LogisticRegression):
            logger.info("Using LinearExplainer for LogisticRegression.")
            # LinearExplainer requires background data; use a zero
            # baseline when the caller has not provided training data.
            n_features = len(self.feature_names)
            background = np.zeros((1, n_features))
            return shap.LinearExplainer(self.model, background)

        # Generic fallback — KernelExplainer is model-agnostic but slow.
        logger.warning(
            "No specialised explainer for '%s'; falling back to "
            "KernelExplainer (may be slow).",
            model_type,
        )
        n_features = len(self.feature_names)
        background = np.zeros((1, n_features))
        return shap.KernelExplainer(self.model.predict_proba, background)

    # ------------------------------------------------------------------
    # Core SHAP computation
    # ------------------------------------------------------------------
    def compute_shap_values(
        self,
        X: np.ndarray | pd.DataFrame,
    ) -> np.ndarray:
        """Compute SHAP values for a feature matrix.

        Parameters
        ----------
        X:
            Feature matrix (n_samples × n_features).

        Returns
        -------
        np.ndarray
            SHAP values array of the same shape as *X*.  For binary
            classifiers the values correspond to the **positive class**
            (churn = 1).
        """
        logger.info("Computing SHAP values for %d sample(s) ...", len(X))

        raw = self._explainer.shap_values(X)

        # TreeExplainer for binary classifiers may return a list of two
        # arrays [class_0, class_1]; we want the positive-class array.
        if isinstance(raw, list):
            return np.asarray(raw[1])

        return np.asarray(raw)

    # ------------------------------------------------------------------
    # Global feature importance
    # ------------------------------------------------------------------
    @staticmethod
    def get_top_features(
        shap_values: np.ndarray,
        feature_names: List[str],
        n: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return the top-*n* features ranked by mean |SHAP value|.

        Parameters
        ----------
        shap_values:
            SHAP value array (n_samples × n_features).
        feature_names:
            Matching feature names.
        n:
            How many features to return.

        Returns
        -------
        list[dict]
            Each dict has ``feature_name`` and ``mean_abs_shap``.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs)[::-1][:n]

        results: List[Dict[str, Any]] = []
        for idx in top_indices:
            results.append({
                "feature_name": feature_names[idx],
                "mean_abs_shap": float(mean_abs[idx]),
            })

        logger.info("Top %d features: %s", n, results)
        return results

    # ------------------------------------------------------------------
    # Single-prediction explanation
    # ------------------------------------------------------------------
    def explain_single_prediction(
        self,
        X_single: np.ndarray | pd.DataFrame,
        feature_names: List[str],
    ) -> List[Dict[str, Any]]:
        """Explain one prediction with the top-3 SHAP drivers.

        Parameters
        ----------
        X_single:
            A single-row feature matrix (1 × n_features).
        feature_names:
            Feature column names.

        Returns
        -------
        list[dict]
            Each dict contains:

            * ``feature_name``  – column name
            * ``shap_value``    – signed SHAP contribution
            * ``feature_value`` – the actual input value
            * ``direction``     – ``'increases'`` or ``'decreases'``
                                  churn risk
            * ``plain_english`` – human-friendly explanation
        """
        shap_vals = self.compute_shap_values(X_single)

        # Ensure 1-D
        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        # Flatten feature values
        if isinstance(X_single, pd.DataFrame):
            feature_values = X_single.iloc[0].values
        else:
            feature_values = np.asarray(X_single).flatten()

        top_3_indices = np.argsort(np.abs(shap_vals))[::-1][:3]

        explanations: List[Dict[str, Any]] = []
        for idx in top_3_indices:
            feat = feature_names[idx]
            sv = float(shap_vals[idx])
            fv = feature_values[idx]
            direction = "increases" if sv > 0 else "decreases"

            # Build plain-English string
            plain = self._render_explanation(feat, fv)

            explanations.append({
                "feature_name": feat,
                "shap_value": sv,
                "feature_value": fv,
                "direction": f"{direction} churn risk",
                "plain_english": plain,
            })

        return explanations

    # ------------------------------------------------------------------
    # Template rendering
    # ------------------------------------------------------------------
    @staticmethod
    def _render_explanation(feature_name: str, value: Any) -> str:
        """Render the plain-English template for *feature_name*.

        Falls back to a generic message when no template is registered.
        """
        # Strip scikit-learn ColumnTransformer prefixes (e.g. num__, cat__, remainder__)
        clean_name = feature_name
        for prefix in ("num__", "cat__", "remainder__"):
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break

        template = FEATURE_EXPLANATIONS.get(clean_name)
        if template is None:
            return (
                f"Feature '{feature_name}' with value {value} contributes "
                f"to the churn prediction."
            )

        try:
            # MonthlyCharges needs a comparison token
            if clean_name == "MonthlyCharges":
                # Rough average from the Telco dataset (~$64.76)
                avg = 64.76
                try:
                    numeric_val = float(value)
                    comparison = "above" if numeric_val > avg else "below"
                except (TypeError, ValueError):
                    comparison = "around"
                return template.format(value=value, comparison=comparison)

            return template.format(value=value)
        except (KeyError, IndexError, ValueError):
            return template  # Return unformatted as last resort

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    def plot_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        max_display: int = 15,
        save_path: Optional[str] = None,
    ) -> None:
        """Create a SHAP beeswarm (summary) plot.

        Parameters
        ----------
        shap_values:
            SHAP values (n_samples × n_features).
        feature_names:
            Feature column names.
        max_display:
            Maximum number of features to display.
        save_path:
            If given, save figure to this path.
        """
        logger.info("Generating SHAP summary plot ...")

        shap.summary_plot(
            shap_values,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP summary plot saved to %s", save_path)

        plt.show()

    def plot_waterfall(
        self,
        shap_values_single: np.ndarray,
        feature_names: List[str],
        save_path: Optional[str] = None,
    ) -> None:
        """Create a SHAP waterfall plot for a single prediction.

        Parameters
        ----------
        shap_values_single:
            1-D SHAP values for one sample.
        feature_names:
            Feature column names.
        save_path:
            If given, save figure to this path.
        """
        logger.info("Generating SHAP waterfall plot ...")

        # Build an Explanation object for the modern SHAP API
        explanation = shap.Explanation(
            values=shap_values_single,
            base_values=self._get_expected_value(),
            feature_names=feature_names,
        )
        shap.plots.waterfall(explanation, show=False)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("SHAP waterfall plot saved to %s", save_path)

        plt.show()

    def _get_expected_value(self) -> float:
        """Retrieve the base (expected) value from the SHAP explainer."""
        ev = self._explainer.expected_value
        # Binary classifiers may return a list; take the positive-class entry.
        if isinstance(ev, (list, np.ndarray)):
            return float(ev[1]) if len(ev) > 1 else float(ev[0])
        return float(ev)


# =========================================================================
# End-to-end prediction + explanation
# =========================================================================

def generate_prediction_explanation(
    model: Any,
    preprocessor: Any,
    raw_input_df: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Produce a churn prediction **and** a plain-English explanation.

    This is the primary entry-point used by the API layer.

    Parameters
    ----------
    model:
        Fitted estimator.
    preprocessor:
        Fitted sklearn transformer / pipeline whose ``transform``
        method converts a raw DataFrame into the model's feature space.
    raw_input_df:
        A single-row DataFrame with the raw customer features
        (before preprocessing).
    feature_names:
        Feature names after preprocessing.  If *None*, attempts to
        infer them from the preprocessor via
        ``get_feature_names_out()``.

    Returns
    -------
    dict
        ``churn_probability``, ``churn_prediction``, ``explanations``
        (list of top-3 SHAP drivers with plain English).
    """
    # --- Resolve feature names -------------------------------------------
    if feature_names is None:
        try:
            feature_names = list(preprocessor.get_feature_names_out())
        except AttributeError:
            raise ValueError(
                "Cannot infer feature names from the preprocessor. "
                "Please pass `feature_names` explicitly."
            )

    # --- Pre-process -----------------------------------------------------
    X_transformed = preprocessor.transform(raw_input_df)

    # --- Predict ---------------------------------------------------------
    churn_prob: float = float(model.predict_proba(X_transformed)[:, 1][0])
    churn_pred: int = int(model.predict(X_transformed)[0])

    # --- Explain ---------------------------------------------------------
    explainer = ChurnExplainer(model, feature_names)
    explanations = explainer.explain_single_prediction(
        X_transformed, feature_names,
    )

    logger.info(
        "Prediction: prob=%.4f, label=%d, top_driver=%s",
        churn_prob,
        churn_pred,
        explanations[0]["feature_name"] if explanations else "N/A",
    )

    return {
        "churn_probability": churn_prob,
        "churn_prediction": churn_pred,
        "explanations": explanations,
    }
