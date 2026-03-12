"""
Shared utility functions for the IOStat project.

Provides reusable helpers for data loading, feature engineering,
model construction, curve smoothing, and interpolated prediction used
across all training and evaluation scripts.
"""

import os
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from config import HYPERPARAMS, ANCHOR_BLOCK_SIZES, FEATURES

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_data(file_path):
    """Load a performance-result file (Excel or CSV) and return a DataFrame."""
    path = Path(file_path)
    if path.suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def read_and_prep_data(file_path, is_test_file=False):
    """
    Load raw benchmark data and prepare it for ML training or testing.

    Parameters
    ----------
    file_path : str
        Path to the Excel or CSV data file.
    is_test_file : bool
        If True, compute ``actual_total_iops`` for evaluation.
        If False, aggregate by configuration and keep the maximum IOPS.

    Returns
    -------
    pd.DataFrame or None
        Preprocessed DataFrame ready for feature engineering.
    """
    print(f"Loading data from: {file_path}")

    # Robust loading: try CSV first, then Excel
    try:
        df = pd.read_csv(
            file_path,
            converters={"job options/bs": str},
            encoding="latin1",
        )
    except Exception:
        try:
            df = pd.read_excel(
                file_path,
                converters={"job options/bs": str},
            )
        except Exception as e:
            print(f"CRITICAL: Could not read file {file_path}. Error: {e}")
            return None

    print("  Load successful.")

    # Basic cleaning
    df["job options/filename"] = df["job options/filename"].fillna("unknown")

    # Feature extraction
    df["block_size_kb"] = (
        df["job options/bs"].str.replace("k", "", regex=False).astype(int)
    )
    df["read_percent"] = pd.to_numeric(
        df["job options/rwmixread"], errors="coerce"
    )

    # Assign read_percent based on workload type
    df.loc[df["job options/rw"] == "randread", "read_percent"] = 100.0
    df.loc[df["job options/rw"] == "randwrite", "read_percent"] = 0.0
    df.loc[
        (df["job options/rw"] == "randrw") & (df["read_percent"].isna()),
        "read_percent",
    ] = 50.0
    df["read_percent"] = df["read_percent"].fillna(100.0)

    if is_test_file:
        # Test data: compute actual total IOPS
        df["actual_total_iops"] = (
            df["read/iops"].fillna(0) + df["write/iops"].fillna(0)
        )
        return df

    # Training data: aggregate to maximum IOPS per configuration
    print("  Aggregating training data (MAX strategy)...")
    df["total_iops"] = df["read/iops"].fillna(0) + df["write/iops"].fillna(0)
    agg_cols = ["job options/filename", "read_percent", "block_size_kb"]
    df_agg = df.groupby(agg_cols, as_index=False)["total_iops"].max()

    df_minimal = df[
        agg_cols + ["job options/rw", "job options/bs"]
    ].drop_duplicates(subset=agg_cols)
    df_final_agg = pd.merge(df_agg, df_minimal, on=agg_cols, how="left")
    return df_final_agg


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df, exclude_sata=False):
    """
    Extract the storage device name from the raw file-path column and
    optionally filter out SATA devices.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a ``job options/filename`` column.
    exclude_sata : bool
        If True, drop rows whose device name is 'sata'.

    Returns
    -------
    pd.DataFrame
        Copy of the input with an added ``device`` column.
    """
    if not isinstance(df, pd.DataFrame):
        return None
    df_out = df.copy()

    df_out["device"] = df_out["job options/filename"].str.extract(
        r"/dev/(\w+)"
    )
    df_out["device"] = df_out["device"].str.replace(
        r"\d+$", "", regex=True
    )
    df_out["device"] = df_out["device"].str.replace(
        r"n\d+$", "", regex=True
    )
    df_out["device"] = df_out["device"].fillna("unknown")

    if exclude_sata:
        df_out = df_out[df_out["device"].str.lower() != "sata"]

    return df_out


def feature_engineering_evaluation(df):
    """
    Prepare evaluation/benchmark data for model prediction.

    Handles both string and numeric block-size columns, computes
    ``read_percent``, and extracts the device identifier.

    Parameters
    ----------
    df : pd.DataFrame
        Raw evaluation data.

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with ``block_size_kb``, ``read_percent``,
        and ``device`` columns.
    """
    df = df.copy()

    # Block size: strip 'k' suffix and convert to numeric
    df["block_size_kb"] = (
        df["job options/bs"]
        .astype(str)
        .str.replace(r"k$", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    df["block_size_kb"] = pd.to_numeric(df["block_size_kb"], errors="coerce")
    df["block_size_kb"] = df["block_size_kb"].astype("Int64")

    # Read/write percentage
    df["read_percent"] = pd.to_numeric(
        df["job options/rwmixread"], errors="coerce"
    )
    df.loc[df["job options/rw"] == "randread", "read_percent"] = 100.0
    df.loc[df["job options/rw"] == "randwrite", "read_percent"] = 0.0
    df.loc[
        (df["job options/rw"] == "randrw") & (df["read_percent"].isna()),
        "read_percent",
    ] = 50.0
    df["read_percent"] = df["read_percent"].fillna(100.0)

    # Device name extraction
    df["device"] = df["job options/filename"].astype(str).str.extract(
        r"/dev/(\w+)"
    )
    df["device"] = df["device"].str.replace(r"\d+$", "", regex=True)
    df["device"] = df["device"].str.replace(r"n\d+$", "", regex=True)
    df["device"] = df["device"].fillna("unknown")

    return df


# ---------------------------------------------------------------------------
# Curve smoothing
# ---------------------------------------------------------------------------

def smooth_curve(x, y, n=300):
    """
    Return a smoothed version of the (x, y) curve using cubic splines.

    Falls back gracefully when fewer than 4 data points are available.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Remove non-finite values
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    if len(x) < 2:
        return x, y

    # Sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    k = min(3, len(x) - 1)
    if k < 2:
        return x, y

    x_new = np.linspace(x.min(), x.max(), n)
    spline = make_interp_spline(x, y, k=k)
    y_new = spline(x_new)
    return x_new, y_new


# ---------------------------------------------------------------------------
# ML pipeline construction
# ---------------------------------------------------------------------------

def get_base_pipelines():
    """
    Build the four standard ML pipelines used across all training scripts.

    Returns
    -------
    dict[str, Pipeline]
        Mapping from model name to an unfitted scikit-learn Pipeline.
    """
    return {
        "LINEAR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "LASSO_POLY": Pipeline([
            ("poly", PolynomialFeatures(
                degree=HYPERPARAMS["lasso_degree"], include_bias=False
            )),
            ("scaler", StandardScaler()),
            ("model", Lasso(
                alpha=HYPERPARAMS["lasso_alpha"], max_iter=10000
            )),
        ]),
        "RANDOM_FOREST": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=HYPERPARAMS["rf_n_estimators"],
                max_depth=HYPERPARAMS["rf_max_depth"],
                min_samples_leaf=HYPERPARAMS["rf_min_samples_leaf"],
                random_state=42,
            )),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf", C=100)),
        ]),
    }


# ---------------------------------------------------------------------------
# Interpolated prediction
# ---------------------------------------------------------------------------

def predict_with_interpolation(model, X_input, anchor_block_sizes=None):
    """
    Predict IOPS using linear interpolation between standard anchor points.

    For block sizes that match an anchor exactly, the model predicts
    directly.  For non-standard block sizes, the prediction is linearly
    interpolated between the two nearest anchors.

    Parameters
    ----------
    model : estimator
        A fitted scikit-learn model or pipeline.
    X_input : array-like, shape (n, 2)
        Columns are ``[read_percent, block_size_kb]``.
    anchor_block_sizes : list[int] or None
        Anchor points for interpolation.  Defaults to
        ``config.ANCHOR_BLOCK_SIZES``.

    Returns
    -------
    np.ndarray
        Predicted values.
    """
    if anchor_block_sizes is None:
        anchor_block_sizes = sorted(ANCHOR_BLOCK_SIZES)

    predictions = []
    for row in X_input:
        r_pct, bs = row[0], row[1]

        if bs in anchor_block_sizes:
            pred = model.predict([[r_pct, bs]])[0]
        else:
            lower_bs = None
            upper_bs = None
            for anchor in anchor_block_sizes:
                if anchor < bs:
                    lower_bs = anchor
                if anchor > bs and upper_bs is None:
                    upper_bs = anchor

            if lower_bs is None or upper_bs is None:
                # Outside anchor range — fall back to direct prediction
                pred = model.predict([[r_pct, bs]])[0]
            else:
                # Linear interpolation between neighbours
                pred_lower = model.predict([[r_pct, lower_bs]])[0]
                pred_upper = model.predict([[r_pct, upper_bs]])[0]
                slope = (pred_upper - pred_lower) / (upper_bs - lower_bs)
                pred = pred_lower + (bs - lower_bs) * slope

        predictions.append(pred)

    return np.array(predictions)


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------

def max_error_index(y_true, y_pred):
    """Return the index of the data point with the largest absolute error."""
    errors = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    return int(np.argmax(errors))
