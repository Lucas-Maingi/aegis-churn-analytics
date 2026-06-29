"""
Churn Prediction — Data Loader
===============================
Handles downloading, caching, schema-validation, and initial cleaning
of the IBM Telco Customer Churn dataset.

Typical usage::

    from churn_prediction.data.loader import load_telco_data, get_feature_target_split

    df = load_telco_data()          # cached after first download
    X, y = get_feature_target_split(df)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.error import URLError

import pandas as pd

from ..config import (
    DATASET_FILENAME,
    DATASET_URLS,
    EXPECTED_COLUMNS,
    ID_COLUMN,
    RAW_DATA_DIR,
    TARGET_COLUMN,
)

logger = logging.getLogger(__name__)

# ── Column groups used during cleaning ──────────────────────────────────────

# Columns where "No internet service" should collapse to "No"
_INTERNET_DEPENDENT_COLS: List[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Column where "No phone service" should collapse to "No"
_PHONE_DEPENDENT_COLS: List[str] = ["MultipleLines"]

# All columns that need the "No *" → "No" replacement
_COLLAPSE_NO_SERVICE_COLS: List[str] = (
    _INTERNET_DEPENDENT_COLS + _PHONE_DEPENDENT_COLS
)


# ── Custom exceptions ──────────────────────────────────────────────────────


class DataLoadError(Exception):
    """Raised when the dataset cannot be downloaded or loaded from disk."""


class SchemaValidationError(Exception):
    """Raised when loaded data does not match the expected column schema."""


class InputValidationError(Exception):
    """Raised when API input data is missing required feature columns."""


# ── Private helpers ─────────────────────────────────────────────────────────


def _download_csv(dest: Path) -> pd.DataFrame:
    """Try each URL in *DATASET_URLS* until a successful download.

    The raw CSV is persisted to *dest* so subsequent calls hit disk.

    Raises
    ------
    DataLoadError
        If every URL fails.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None
    for url in DATASET_URLS:
        try:
            logger.info("Downloading dataset from %s ...", url)
            df = pd.read_csv(url)
            # Persist for future offline use
            df.to_csv(dest, index=False)
            logger.info(
                "Downloaded and cached dataset (%d rows x %d cols) -> %s",
                len(df),
                len(df.columns),
                dest,
            )
            return df
        except (URLError, OSError, pd.errors.ParserError) as exc:
            logger.warning("Download failed for %s: %s", url, exc)
            last_error = exc

    raise DataLoadError(
        f"All download URLs failed. Last error: {last_error}"
    )


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert that *df* contains every column listed in *EXPECTED_COLUMNS*.

    Raises
    ------
    SchemaValidationError
        With details about missing columns.
    """
    actual = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing = expected - actual
    if missing:
        raise SchemaValidationError(
            f"Dataset is missing {len(missing)} expected column(s): "
            f"{sorted(missing)}"
        )
    logger.debug("Schema validation passed — all %d columns present.", len(expected))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply initial, non-destructive cleaning to the raw dataset.

    Steps
    -----
    1. Strip whitespace from all object (string) columns.
    2. Convert ``TotalCharges`` from string → float, filling blanks with 0.
    3. Map ``Churn`` from Yes/No to 1/0.
    4. Replace *"No internet service"* and *"No phone service"* with *"No"*.

    Note: ``customerID`` is deliberately kept — the preprocessor drops it.
    """
    df = df.copy()

    # 1. Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    # 2. TotalCharges → float (11 blanks become NaN → 0.0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blanks = df["TotalCharges"].isna().sum()
    if n_blanks > 0:
        logger.info(
            "Filled %d blank TotalCharges values with 0.0.", n_blanks
        )
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 3. Churn → binary int
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})
    if df[TARGET_COLUMN].isna().any():
        n_bad = df[TARGET_COLUMN].isna().sum()
        logger.warning(
            "%d rows have unexpected Churn values (not Yes/No); "
            "dropping them.",
            n_bad,
        )
        df = df.dropna(subset=[TARGET_COLUMN])
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # 4. Collapse 'No internet service' / 'No phone service' → 'No'
    for col in _COLLAPSE_NO_SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace(
                {"No internet service": "No", "No phone service": "No"}
            )

    logger.info(
        "Cleaning complete - %d rows x %d cols, churn rate %.1f%%.",
        len(df),
        len(df.columns),
        df[TARGET_COLUMN].mean() * 100,
    )
    return df


# ── Public API ──────────────────────────────────────────────────────────────


def load_telco_data(
    data_dir: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load the IBM Telco Customer Churn dataset.

    On the first call the CSV is downloaded from a public URL and cached
    to ``data/raw/``.  Subsequent calls read from the local cache.

    Parameters
    ----------
    data_dir : Path, optional
        Override the default raw-data directory (``data/raw/``).
    force_download : bool, default False
        Re-download even if the local file already exists.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with 21 original columns.  ``TotalCharges`` is
        float, ``Churn`` is int (1/0), and *"No internet/phone service"*
        values have been collapsed to *"No"*.

    Raises
    ------
    DataLoadError
        If all download URLs fail and no local cache exists.
    SchemaValidationError
        If the loaded CSV is missing one or more expected columns.
    """
    dest_dir = data_dir or RAW_DATA_DIR
    csv_path = dest_dir / DATASET_FILENAME

    if csv_path.exists() and not force_download:
        logger.info("Loading cached dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
    else:
        df = _download_csv(csv_path)

    _validate_schema(df)
    df = _clean(df)
    return df


def get_feature_target_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a cleaned DataFrame into features and target.

    ``customerID`` is dropped from the feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`load_telco_data`.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except ``customerID`` and ``Churn``).
    y : pd.Series
        Binary target vector (1 = churned, 0 = retained).
    """
    cols_to_drop = [
        c for c in [ID_COLUMN, TARGET_COLUMN] if c in df.columns
    ]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COLUMN].copy()
    logger.debug(
        "Feature-target split: X %s, y %s (churn rate %.1f%%).",
        X.shape,
        y.shape,
        y.mean() * 100,
    )
    return X, y


def drop_customer_id(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* without the ``customerID`` column.

    Silently returns the original DataFrame if the column is absent.
    """
    if ID_COLUMN in df.columns:
        return df.drop(columns=[ID_COLUMN])
    return df


# ── Required feature columns for API input validation ──────────────────────

_REQUIRED_FEATURE_COLUMNS: List[str] = [
    c for c in EXPECTED_COLUMNS if c not in (ID_COLUMN, TARGET_COLUMN)
]


def validate_input_data(df: pd.DataFrame) -> List[str]:
    """Check that a DataFrame from API input has all required feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data to validate (e.g. from a JSON request body).

    Returns
    -------
    list of str
        Empty list if validation passes; otherwise a list of human-readable
        error messages.

    Raises
    ------
    InputValidationError
        If any required feature columns are missing.  The exception message
        enumerates the missing columns.

    Examples
    --------
    >>> errors = validate_input_data(api_df)
    >>> if errors:
    ...     return {"errors": errors}, 400
    """
    errors: List[str] = []

    missing = set(_REQUIRED_FEATURE_COLUMNS) - set(df.columns)
    if missing:
        errors.append(
            f"Missing required feature column(s): {sorted(missing)}"
        )

    # Check for completely empty input
    if df.empty:
        errors.append("Input DataFrame is empty (0 rows).")

    if errors:
        raise InputValidationError("; ".join(errors))

    logger.debug("Input validation passed — %d rows, %d columns.", len(df), len(df.columns))
    return errors
