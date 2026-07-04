"""
Churn Prediction — Feature Engineering & Preprocessing
=======================================================
Wraps scikit-learn's ``ColumnTransformer`` / ``Pipeline`` into a
serialisable :class:`ChurnPreprocessor` class that handles:

* Engineered features (tenure buckets, interaction terms, aggregates)
* Binary Yes/No → 1/0 mapping
* Standard-scaling of numerics
* One-hot encoding of multi-class categoricals

Typical usage::

    from churn_prediction.data.preprocessor import ChurnPreprocessor

    preprocessor = ChurnPreprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)

    # Serialise
    preprocessor.save("models/preprocessor.joblib")
    loaded = ChurnPreprocessor.load("models/preprocessor.joblib")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import (
    BINARY_CATEGORICAL_FEATURES,
    MULTI_CLASS_CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    PROTECTION_COLUMNS,
    SERVICE_COLUMNS,
    TENURE_BINS,
    TENURE_LABELS,
)

logger = logging.getLogger(__name__)

# ── Contract encoding map (used for the interaction feature) ────────────────

_CONTRACT_ENCODING: Dict[str, int] = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2,
}

# Binary string columns that need Yes/No → 1/0 mapping.
# SeniorCitizen is already int, so it's excluded from this list.
_BINARY_STRING_COLS: List[str] = [
    col for col in BINARY_CATEGORICAL_FEATURES if col != "SeniorCitizen"
]

# ── Engineered-feature column names ────────────────────────────────────────

_ENG_NUMERIC: List[str] = [
    "avg_monthly_charge",
    "num_services",
    "contract_charge_interaction",
]
_ENG_BINARY: List[str] = ["has_protection_bundle"]
_ENG_MULTICLASS: List[str] = ["tenure_bucket"]

# Final column lists fed to the ColumnTransformer
_CT_NUMERIC: List[str] = NUMERICAL_FEATURES + _ENG_NUMERIC
_CT_BINARY: List[str] = BINARY_CATEGORICAL_FEATURES + _ENG_BINARY
_CT_MULTICLASS: List[str] = MULTI_CLASS_CATEGORICAL_FEATURES + _ENG_MULTICLASS


# ── Private helpers ─────────────────────────────────────────────────────────


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features to a copy of *df*.

    This runs **before** the sklearn ``ColumnTransformer`` so that the
    new columns are available for scaling / encoding.

    Features created
    ----------------
    tenure_bucket : str
        Categorical bin label for ``tenure`` (0-12 / 13-24 / 25-48 / 49-72).
    avg_monthly_charge : float
        ``TotalCharges / max(tenure, 1)`` — average revenue per month.
    num_services : int
        Count of *"Yes"* values across the service add-on columns.
        For ``InternetService``, any value other than ``"No"`` is counted.
    has_protection_bundle : int
        1 if the customer subscribes to *all three* protection services
        (OnlineSecurity, TechSupport, DeviceProtection), else 0.
    contract_charge_interaction : float
        Numeric contract code (0 / 1 / 2) × ``MonthlyCharges``.
    """
    df = df.copy()

    # 1. Tenure bucket --------------------------------------------------
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
    ).astype(str)  # keep as plain string for OHE

    # 2. Average monthly charge -----------------------------------------
    df["avg_monthly_charge"] = df["TotalCharges"] / df["tenure"].clip(lower=1)

    # 3. Number of subscribed services ----------------------------------
    def _count_services(row: pd.Series) -> int:
        count = 0
        for col in SERVICE_COLUMNS:
            val = row.get(col, "No")
            # InternetService: anything other than "No" counts
            if col == "InternetService":
                count += int(val != "No")
            else:
                count += int(val == "Yes")
        return count

    df["num_services"] = df.apply(_count_services, axis=1)

    # 4. Protection bundle flag -----------------------------------------
    df["has_protection_bundle"] = (
        (df[PROTECTION_COLUMNS] == "Yes").all(axis=1).astype(int)
    )

    # 5. Contract × MonthlyCharges interaction --------------------------
    df["contract_charge_interaction"] = (
        df["Contract"].map(_CONTRACT_ENCODING).fillna(0).astype(float)
        * df["MonthlyCharges"]
    )

    logger.debug(
        "Engineered 5 features — DataFrame now has %d columns.", len(df.columns)
    )
    return df


def _encode_binary_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Map Yes/No string columns to 1/0 ints in-place.

    ``SeniorCitizen`` is already numeric and is left untouched.
    ``gender`` is mapped as Female=1, Male=0 to keep it binary.
    """
    df = df.copy()

    # Standard Yes/No mapping
    yes_no_map = {"Yes": 1, "No": 0}
    for col in _BINARY_STRING_COLS:
        if col not in df.columns:
            continue
        if col == "gender":
            df[col] = df[col].map({"Female": 1, "Male": 0})
        # Guard against double-mapping columns that are already numeric. We test
        # for a non-numeric dtype rather than `== object` because pandas 3.0
        # infers string columns as StringDtype (not object), which would
        # otherwise skip the mapping and leak raw "Yes"/"No" values downstream.
        elif not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(yes_no_map)

    # Make sure SeniorCitizen stays int
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


# ── Public class ────────────────────────────────────────────────────────────


class ChurnPreprocessor:
    """End-to-end feature-engineering + sklearn preprocessing wrapper.

    The preprocessor is split into two stages:

    1. **Feature engineering** – pure-pandas transformations
       (:func:`_engineer_features`, :func:`_encode_binary_strings`).
    2. **ColumnTransformer** – standard-scales numerics, passes through
       binary columns, and one-hot-encodes multi-class categoricals.

    The fitted ``ColumnTransformer`` is stored as :attr:`pipeline_` and
    can be serialised / deserialised with :meth:`save` / :meth:`load`.

    Parameters
    ----------
    None

    Attributes
    ----------
    pipeline_ : sklearn.compose.ColumnTransformer
        Fitted column transformer (available after :meth:`fit`).
    is_fitted_ : bool
        Whether :meth:`fit` has been called.
    """

    def __init__(self) -> None:
        self.pipeline_: Optional[ColumnTransformer] = None
        self.is_fitted_: bool = False

    # ── Build the sklearn ColumnTransformer ─────────────────────────────

    @staticmethod
    def _build_column_transformer() -> ColumnTransformer:
        """Construct an *unfitted* ``ColumnTransformer``.

        Transformer layout
        -------------------
        ``num``   : StandardScaler   → numeric features
        ``bin``   : passthrough       → binary features (already 0/1)
        ``cat``   : OneHotEncoder     → multi-class categoricals
        """
        return ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    _CT_NUMERIC,
                ),
                (
                    "bin",
                    "passthrough",
                    _CT_BINARY,
                ),
                (
                    "cat",
                    OneHotEncoder(
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    _CT_MULTICLASS,
                ),
            ],
            remainder="drop",  # drops customerID & anything unexpected
            verbose_feature_names_out=True,
        )

    # ── Core API ────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame) -> "ChurnPreprocessor":
        """Fit the preprocessor on training features.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix (*without* the target column).

        Returns
        -------
        self
        """
        X_eng = _engineer_features(X)
        X_enc = _encode_binary_strings(X_eng)

        self.pipeline_ = self._build_column_transformer()
        self.pipeline_.fit(X_enc)
        self.is_fitted_ = True

        n_features = len(self.get_feature_names())
        logger.info(
            "ChurnPreprocessor fitted - %d input cols -> %d output features.",
            len(X.columns),
            n_features,
        )
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using the fitted pipeline.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix (same schema as training data).

        Returns
        -------
        np.ndarray
            2-D array of transformed features, shape ``(n_samples, n_features)``.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called yet.
        """
        if not self.is_fitted_ or self.pipeline_ is None:
            raise RuntimeError(
                "ChurnPreprocessor has not been fitted. Call fit() first."
            )

        X_eng = _engineer_features(X)
        X_enc = _encode_binary_strings(X_eng)
        return self.pipeline_.transform(X_enc)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Convenience method: fit then transform in a single call.

        Parameters
        ----------
        X : pd.DataFrame
            Raw feature matrix.

        Returns
        -------
        np.ndarray
            Transformed feature array.
        """
        return self.fit(X).transform(X)

    # ── Feature introspection ───────────────────────────────────────────

    def get_feature_names(self) -> List[str]:
        """Return the ordered list of output feature names.

        The names are extracted from the fitted ``ColumnTransformer``
        via ``get_feature_names_out()``.

        Returns
        -------
        list of str

        Raises
        ------
        RuntimeError
            If the preprocessor has not been fitted.
        """
        if not self.is_fitted_ or self.pipeline_ is None:
            raise RuntimeError(
                "ChurnPreprocessor has not been fitted. "
                "Call fit() before get_feature_names()."
            )
        return list(self.pipeline_.get_feature_names_out())

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Union[str, Path]) -> Path:
        """Serialise the fitted preprocessor to disk with joblib.

        Parameters
        ----------
        path : str or Path
            Destination file path (e.g. ``models/preprocessor.joblib``).

        Returns
        -------
        Path
            Resolved path where the file was written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Saved ChurnPreprocessor → %s", path)
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ChurnPreprocessor":
        """Deserialise a previously saved preprocessor.

        Parameters
        ----------
        path : str or Path
            Path to a ``.joblib`` file created by :meth:`save`.

        Returns
        -------
        ChurnPreprocessor

        Raises
        ------
        FileNotFoundError
            If *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Loaded object is {type(obj).__name__}, "
                f"expected {cls.__name__}."
            )
        logger.info("Loaded ChurnPreprocessor ← %s", path)
        return obj

    # ── Repr ────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        n_feats = (
            len(self.get_feature_names()) if self.is_fitted_ else "?"
        )
        return (
            f"ChurnPreprocessor(status={status}, "
            f"output_features={n_feats})"
        )
