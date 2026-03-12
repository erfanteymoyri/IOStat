"""
Mentor's visualization with block-size generalization logic.

Unified data from a single file; trains on powers-of-2 block sizes,
tests on non-standard block sizes.  Excludes SATA devices.  Uses the
mentor's 2-panel visualization with larger font sizes.

Usage
-----
    python 12_mentor_generalization_viz.py <data_path>
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    FONT_SIZE_LABEL_MENTOR, FONT_SIZE_TICK_MENTOR, FONT_SIZE_LEGEND_MENTOR,
    HYPERPARAMS, MODEL_ORDER, HATCH_MAP, BAR_COLORS, LEGEND_LABELS,
    ERROR_PANEL_RATIO, EXPORT_PDF, FEATURES, STANDARD_BLOCK_SIZES,
)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
PLOT_DIR = "barcharts_mentor_generalization"
PLOT_EXT = "pdf"


# ------------------------------------------------------------------ #
# Data loading
# ------------------------------------------------------------------ #
def load_unified_data(path):
    """Load and aggregate unified data (MAX strategy per config)."""
    print(f"Loading data from: {path}")
    try:
        df = pd.read_csv(path, converters={"job options/bs": str}, encoding="latin1")
    except Exception:
        df = pd.read_excel(path, converters={"job options/bs": str})

    df["job options/filename"] = df["job options/filename"].fillna("unknown")
    df["block_size_kb"] = df["job options/bs"].str.replace("k", "").astype(int)

    df["read_percent"] = pd.to_numeric(df["job options/rwmixread"], errors="coerce")
    df.loc[df["job options/rw"] == "randread", "read_percent"] = 100
    df.loc[df["job options/rw"] == "randwrite", "read_percent"] = 0
    df.loc[(df["job options/rw"] == "randrw") & df["read_percent"].isna(), "read_percent"] = 50
    df["read_percent"] = df["read_percent"].fillna(100)

    df["total_iops"] = df["read/iops"].fillna(0) + df["write/iops"].fillna(0)

    print("Aggregating data (MAX strategy per config)...")
    agg_cols = ["job options/filename", "read_percent", "block_size_kb"]
    df_agg = df.groupby(agg_cols, as_index=False)["total_iops"].max()
    meta = df[agg_cols + ["job options/rw", "job options/bs"]].drop_duplicates()
    return pd.merge(df_agg, meta, on=agg_cols, how="left")


def engineer_features(df):
    """Extract device and exclude sata devices."""
    df = df.copy()
    df["device"] = df["job options/filename"].str.extract(r"/dev/(\w+)")
    df["device"] = df["device"].str.replace(r"\d+$", "", regex=True)
    df["device"] = df["device"].fillna("unknown")
    df = df[df["device"].str.lower() != "sata"]
    return df


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #
def train_models(df, features):
    """Train all four model families per device."""
    print("\n--- Starting Model Training ---")
    models = {}
    for dev in df["device"].unique():
        if dev == "unknown":
            continue

        sub = df[df["device"] == dev]
        if len(sub) < 3:
            continue

        X = sub[features]
        y = sub["total_iops"]

        models[dev] = {
            "LINEAR": Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
            "LASSO_POLY": Pipeline([
                ("p", PolynomialFeatures(HYPERPARAMS["lasso_degree"], include_bias=False)),
                ("s", StandardScaler()),
                ("m", Lasso(alpha=HYPERPARAMS["lasso_alpha"], max_iter=10000)),
            ]),
            "RANDOM_FOREST": Pipeline([
                ("s", StandardScaler()),
                ("m", RandomForestRegressor(
                    n_estimators=HYPERPARAMS["rf_n_estimators"],
                    max_depth=HYPERPARAMS["rf_max_depth"],
                    min_samples_leaf=HYPERPARAMS["rf_min_samples_leaf"],
                    random_state=42)),
            ]),
            "SVR": Pipeline([("s", StandardScaler()), ("m", SVR(C=100))]),
        }

        for m in models[dev].values():
            m.fit(X, y)
    return models


# ------------------------------------------------------------------ #
# Prediction
# ------------------------------------------------------------------ #
def evaluate_and_predict(df, models, features):
    """Predict on test data using per-device models."""
    print("\n--- Starting Prediction on Test Data ---")
    out = []
    for dev, dev_models in models.items():
        sub = df[df["device"] == dev]
        if sub.empty:
            continue

        X = sub[features]
        res = sub[["device", "read_percent", "block_size_kb"]].copy()
        res["Actual"] = sub["total_iops"]

        for name, model in dev_models.items():
            pred = model.predict(X)
            pred[pred < 0] = 0
            res[name] = pred

        out.append(res)
    return pd.concat(out, ignore_index=True) if out else None


# ------------------------------------------------------------------ #
# Visualization – mentor's 2-panel grid
# ------------------------------------------------------------------ #
def create_plots(df, error_line_color="navy", error_line_width=2.5,
                 error_line_alpha=0.9):
    """Two-panel plot with mentor-style large fonts."""
    print("\n--- Generating Detailed Grid Plots ---")
    os.makedirs(PLOT_DIR, exist_ok=True)

    for (dev, bs), g in df.groupby(["device", "block_size_kb"]):
        g = g.sort_values("read_percent")
        reads = g["read_percent"].values
        x = np.arange(len(reads))

        fig, (ax_err, ax_bar) = plt.subplots(
            2, 1, figsize=(10, 4), sharex=True,
            gridspec_kw={"height_ratios": [ERROR_PANEL_RATIO, 2.0]},
        )

        bar_w = 0.18
        offsets = (np.arange(len(MODEL_ORDER)) - 2) * bar_w

        # Error panel
        ax_err.axhline(0, color="black", linestyle="--", linewidth=1)
        ax_err.set_ylabel("    Error (%)", fontsize=FONT_SIZE_LABEL_MENTOR)
        ax_err.tick_params(axis="y", labelsize=FONT_SIZE_TICK_MENTOR)
        ax_err.yaxis.tick_left()
        ax_err.yaxis.set_label_position("left")

        for wi in range(len(reads)):
            row = g.iloc[wi]
            errors = []
            for m in MODEL_ORDER:
                if m == "Actual":
                    errors.append(0)
                else:
                    a, p = row["Actual"], row[m]
                    errors.append((p - a) / a * 100 if a > 0 else 0)

            ax_err.plot(x[wi] + offsets, errors, marker="o",
                        linewidth=error_line_width, alpha=error_line_alpha,
                        color=error_line_color)

        ax_err.grid(axis="y", linestyle="--", alpha=0.4)

        # Bar panel
        for i, m in enumerate(MODEL_ORDER):
            ax_bar.bar(x + offsets[i], g[m].values / 1000, width=bar_w,
                       color=BAR_COLORS[m], hatch=HATCH_MAP[m],
                       edgecolor="black")

        ax_bar.set_xlabel("Read %", fontsize=FONT_SIZE_LABEL_MENTOR)
        ax_bar.set_ylabel("IOPS (x1000)", fontsize=FONT_SIZE_LABEL_MENTOR)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(reads, fontsize=FONT_SIZE_TICK_MENTOR)
        ax_bar.tick_params(axis="y", labelsize=FONT_SIZE_TICK_MENTOR)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

        legend_handles = [
            mpatches.Patch(facecolor=BAR_COLORS[m], hatch=HATCH_MAP[m],
                           edgecolor="black", label=LEGEND_LABELS[m])
            for m in MODEL_ORDER
        ]
        fig.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.53, 1.06), ncol=5,
                   fontsize=FONT_SIZE_LEGEND_MENTOR, frameon=True,
                   handlelength=1.4, handletextpad=0.4,
                   columnspacing=0.8, labelspacing=0.3, borderpad=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        if EXPORT_PDF:
            fname = f"{PLOT_DIR}/bar_error_grid_{dev}_{bs}k.{PLOT_EXT}"
            plt.savefig(fname, bbox_inches="tight")
        plt.close()


# ------------------------------------------------------------------ #
# Summary MAPE plot (mentor style)
# ------------------------------------------------------------------ #
def create_summary_plot(results_df, bar_width=0.24):
    """Grouped bar chart of MAPE per device with mentor fonts."""
    print("\n--- Generating Final Summary Plot ---")

    model_cols = [m for m in MODEL_ORDER if m != "Actual"]
    device_labels = {
        "nvme0n": "NVMe", "nvme1n": "Optane",
        "pmem": "Pmem", "sda": "Sata SSD",
    }

    summary = []
    for m in model_cols:
        tmp = results_df[results_df["Actual"] > 0].copy()
        tmp["APE"] = np.abs((tmp[m] - tmp["Actual"]) / tmp["Actual"]) * 100
        s = tmp.groupby("device")["APE"].mean().reset_index()
        s["Model"] = m
        summary.append(s)

    plot_df = pd.concat(summary, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    devices = plot_df["device"].unique()
    x = np.arange(len(devices))
    n_models = len(model_cols)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    for i, m in enumerate(model_cols):
        vals = plot_df[plot_df["Model"] == m]["APE"].values
        bars = ax.bar(x + offsets[i], vals, width=bar_width,
                      color=BAR_COLORS[m], hatch=HATCH_MAP[m],
                      edgecolor="black")
        for b in bars:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width() / 2,
                    h + ax.get_ylim()[1] * 0.015, f"{h:.0f}%",
                    ha="center", va="bottom",
                    fontsize=FONT_SIZE_TICK_MENTOR - 4)

    ax.set_xticks(x)
    ax.set_xticklabels([device_labels.get(d, d) for d in devices],
                       fontsize=FONT_SIZE_TICK_MENTOR)
    ax.set_ylabel("Mean Absolute \nPercentage Error (%)",
                  fontsize=FONT_SIZE_LABEL_MENTOR)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK_MENTOR)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor=BAR_COLORS[m], hatch=HATCH_MAP[m],
                       edgecolor="black", label=LEGEND_LABELS[m])
        for m in model_cols
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.28), ncol=4,
              fontsize=FONT_SIZE_LEGEND_MENTOR,
              handlelength=1.3, handletextpad=0.4,
              columnspacing=0.8, labelspacing=0.3, borderpad=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fname = f"{PLOT_DIR}/Overall_Error.{PLOT_EXT}"
    if EXPORT_PDF:
        plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {fname}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Mentor's visualization with block-size generalization.",
    )
    parser.add_argument("data_path",
                        help="Path to unified data file (e.g. AllDevices2.xlsx)")
    args = parser.parse_args()

    # 1. Load data
    df_raw = load_unified_data(args.data_path)
    if df_raw is None:
        return

    # 2. Feature engineering (excludes sata)
    df_eng = engineer_features(df_raw)

    # 3. Split: train on powers-of-2, test on non-standard block sizes
    print("Splitting Data based on Block Size (Train=Powers of 2, Test=Others)...")
    train_df = df_eng[df_eng["block_size_kb"].isin(STANDARD_BLOCK_SIZES)].copy()
    test_df = df_eng[~df_eng["block_size_kb"].isin(STANDARD_BLOCK_SIZES)].copy()

    print(f"  - Train Data Shape: {train_df.shape}")
    print(f"  - Test Data Shape: {test_df.shape}")

    if test_df.empty:
        print("CRITICAL: No testing data found (no block sizes other than powers of 2). Exiting.")
        return

    # 4. Train
    models = train_models(train_df, FEATURES)

    # 5. Predict
    res = evaluate_and_predict(test_df, models, FEATURES)

    if res is not None:
        agg = res.groupby(
            ["device", "block_size_kb", "read_percent"], as_index=False,
        ).mean(numeric_only=True)

        agg.to_excel("predictions_mentor_generalization.xlsx", index=False)
        create_plots(agg)
        create_summary_plot(agg)
        print("\nScript Completed Successfully!")


if __name__ == "__main__":
    main()
