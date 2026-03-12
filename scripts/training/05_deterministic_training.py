"""
Deterministic training and evaluation.

Trains on AllDevices.xlsx (max IOPS per configuration) and tests on
limit100.xlsx.  Uses four models (LINEAR, LASSO_POLY, RANDOM_FOREST,
SVR).  Produces bar charts comparing Actual vs Predicted IOPS and a
summary MAPE plot.  Results are also exported to Excel.

Usage
-----
    python 05_deterministic_training.py <train_data_path> <test_data_path>
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.base import clone

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    HYPERPARAMS, MODEL_ORDER, HATCH_MAP, BAR_COLORS, LEGEND_LABELS,
    FEATURES,
)
from utils import read_and_prep_data, engineer_features, get_base_pipelines

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
PLOT_DIR = "barcharts_ordinary"
PLOT_FORMAT = "pdf"


# ------------------------------------------------------------------ #
# Training
# ------------------------------------------------------------------ #
def train_models(df_train, features):
    """Train one set of models per device."""
    print("\n--- Starting Model Training ---")
    devices = df_train["device"].unique()
    models = {}

    for dev in devices:
        if dev == "unknown":
            continue
        print(f"Training models for device: {dev}")

        dev_data = df_train[df_train["device"] == dev].copy()
        if len(dev_data) < 5:
            continue

        X_train = dev_data[features]
        y_train = dev_data["total_iops"]

        dev_models = {name: clone(pipe) for name, pipe in get_base_pipelines().items()}
        for model in dev_models.values():
            model.fit(X_train, y_train)

        models[dev] = dev_models
    return models


# ------------------------------------------------------------------ #
# Prediction
# ------------------------------------------------------------------ #
def evaluate_and_predict(df_test, models, features):
    """Predict on the test set using the per-device trained models."""
    print("\n--- Starting Prediction on Test Data ---")
    all_results = []

    for dev, dev_models in models.items():
        dev_test = df_test[df_test["device"] == dev].copy()
        if dev_test.empty:
            continue

        X_test = dev_test[features]
        y_actual = dev_test["actual_total_iops"]

        dev_results = dev_test[["device", "read_percent", "block_size_kb"]].copy()
        dev_results["Actual"] = y_actual

        for name, model in dev_models.items():
            y_pred = model.predict(X_test)
            if name == "LINEAR":
                y_pred[y_pred < 0] = 0
            dev_results[name] = y_pred

        all_results.append(dev_results)

    if not all_results:
        return None
    return pd.concat(all_results, ignore_index=True)


# ------------------------------------------------------------------ #
# Visualization – per-group bar charts
# ------------------------------------------------------------------ #
def create_plots(results_df):
    """Generate bar charts for every (device, block_size) combination."""
    print("\n--- Generating Detailed Bar Charts ---")
    if results_df is None or results_df.empty:
        return

    os.makedirs(PLOT_DIR, exist_ok=True)

    model_cols = [c for c in results_df.columns
                  if c not in ("device", "block_size_kb", "read_percent", "Actual")]
    hue_order = ["Actual"] + model_cols

    colors_list = ["#33a02c", "#1f78b4", "#a6cee3", "#e31a1c", "#fdbf6f", "#6a3d9a"]
    model_colors = {m: colors_list[i % len(colors_list)] for i, m in enumerate(hue_order)}

    for (dev, bs), group_data in results_df.groupby(["device", "block_size_kb"]):
        # MAPE strings for the title
        mape_strs = []
        for m in model_cols:
            y_true = group_data["Actual"]
            y_pred = group_data[m]
            mask = y_true > 0
            if mask.any():
                mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
                mape_strs.append(f"{m}: {mape:.2f}%")
            else:
                mape_strs.append(f"{m}: N/A")
        title_mape = " | ".join(mape_strs)

        data_long = group_data.melt(
            id_vars=["read_percent"], value_vars=hue_order,
            var_name="Model", value_name="IOPS",
        )

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(data=data_long, x="read_percent", y="IOPS",
                         hue="Model", palette=model_colors)

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.25)

        actual_lookup = group_data.set_index("read_percent")["Actual"].to_dict()
        sorted_rp = sorted(group_data["read_percent"].unique())
        n_groups = len(sorted_rp)

        for i, bar in enumerate(ax.patches):
            model_idx = i // n_groups
            group_idx = i % n_groups
            if model_idx >= len(hue_order):
                continue
            model_name = hue_order[model_idx]
            if model_name == "Actual":
                continue

            current_rp = sorted_rp[group_idx]
            actual_val = actual_lookup.get(current_rp, 0)
            pred_val = bar.get_height()

            if actual_val > 0:
                err = (pred_val - actual_val) / actual_val * 100
                lbl = f"{err:+.2f}%"
            else:
                lbl = "N/A"

            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.02,
                    lbl, ha="center", va="bottom", fontsize=8, rotation=90)

        plt.title(f"Device: {dev} | BS: {bs}k\nMAPE: {title_mape}",
                  fontsize=FONT_SIZE_TITLE)
        plt.xlabel("Read %", fontsize=FONT_SIZE_LABEL)
        plt.ylabel("IOPS", fontsize=FONT_SIZE_LABEL)
        plt.tick_params(labelsize=FONT_SIZE_TICK)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=FONT_SIZE_LEGEND)
        plt.tight_layout()

        fname = f"{PLOT_DIR}/bar_{dev}_{bs}k.{PLOT_FORMAT}"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()


# ------------------------------------------------------------------ #
# Visualization – summary MAPE plot
# ------------------------------------------------------------------ #
def create_summary_plot(results_df):
    """Average MAPE per device, grouped by model."""
    print("\n--- Generating Final Summary Plot ---")
    if results_df is None or results_df.empty:
        return

    model_cols = [c for c in results_df.columns
                  if c not in ("device", "block_size_kb", "read_percent", "Actual")]

    summary_data = []
    for m in model_cols:
        valid = results_df[results_df["Actual"] > 0].copy()
        valid["APE"] = np.abs((valid[m] - valid["Actual"]) / valid["Actual"]) * 100
        device_mape = valid.groupby("device")["APE"].mean().reset_index()
        device_mape["Model"] = m
        summary_data.append(device_mape)

    if not summary_data:
        return

    plot_df = pd.concat(summary_data, ignore_index=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=plot_df, x="device", y="APE", hue="Model",
                     palette="viridis")

    plt.title("Final Summary: Average MAPE by Device", fontsize=FONT_SIZE_TITLE)
    plt.ylabel("Mean Absolute Percentage Error (%)", fontsize=FONT_SIZE_LABEL)
    plt.xlabel("Device", fontsize=FONT_SIZE_LABEL)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left",
               fontsize=FONT_SIZE_LEGEND)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for p in ax.patches:
        if p.get_height() > 0:
            ax.annotate(
                f"{p.get_height():.2f}%",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center", va="bottom", fontsize=9,
                xytext=(0, 5), textcoords="offset points",
            )

    fname = f"{PLOT_DIR}/FINAL_SUMMARY_MAPE.{PLOT_FORMAT}"
    plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {fname}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Deterministic ML training & evaluation for IOPS prediction.",
    )
    parser.add_argument("train_path", help="Path to the training data file (e.g. AllDevices.xlsx)")
    parser.add_argument("test_path", help="Path to the test data file (e.g. limit100.xlsx)")
    args = parser.parse_args()

    # 1. Load data
    train_df = read_and_prep_data(args.train_path, is_test_file=False)
    test_df = read_and_prep_data(args.test_path, is_test_file=True)
    if train_df is None or test_df is None:
        return

    # 2. Feature engineering
    df_train_eng = engineer_features(train_df)
    df_test_eng = engineer_features(test_df)

    # 3. Train
    models = train_models(df_train_eng, FEATURES)

    # 4. Predict
    results = evaluate_and_predict(df_test_eng, models, FEATURES)

    if results is not None:
        agg = results.groupby(
            ["device", "block_size_kb", "read_percent"], as_index=False,
        ).mean(numeric_only=True)

        agg.to_excel("predictions_ordinary.xlsx", index=False)
        create_plots(agg)
        create_summary_plot(agg)
        print("\nScript Completed Successfully!")


if __name__ == "__main__":
    main()
