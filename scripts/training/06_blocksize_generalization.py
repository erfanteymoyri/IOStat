"""
Block-size generalization with regime-based training.

Train on standard block sizes (powers of 2: 2, 4, 8, 16, 32, 64, 128)
and test on non-standard block sizes.  Uses a regime-based model
selection where separate models are trained for small (BS <= 16 KB) and
large (BS > 16 KB) regimes.  Excludes SATA devices.

Usage
-----
    python 06_blocksize_generalization.py <data_path>
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    MODEL_ORDER, HATCH_MAP, BAR_COLORS, LEGEND_LABELS,
    ERROR_PANEL_RATIO, EXPORT_PDF, FEATURES, STANDARD_BLOCK_SIZES,
)
from utils import read_and_prep_data, engineer_features, get_base_pipelines

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
PLOT_DIR = "barcharts_regime_based4"
PLOT_EXT = "pdf"
REGIME_THRESHOLD = 16  # small <= 16 KB, large > 16 KB


# ------------------------------------------------------------------ #
# Regime-based training
# ------------------------------------------------------------------ #
def train_models(df_train, features):
    """Train separate model sets for small-BS and large-BS regimes."""
    print("\n--- Starting Regime-Based Model Training ---")
    devices = df_train["device"].unique()
    models = {}

    for dev in devices:
        if dev == "unknown":
            continue
        print(f"Training models for device: {dev}")

        dev_data = df_train[df_train["device"] == dev].copy()
        dev_small = dev_data[dev_data["block_size_kb"] <= REGIME_THRESHOLD]
        dev_large = dev_data[dev_data["block_size_kb"] > REGIME_THRESHOLD]

        models[dev] = {"small": {}, "large": {}}

        if len(dev_small) >= 3:
            pipes_small = get_base_pipelines()
            X_small = dev_small[features]
            y_small = dev_small["total_iops"]
            for pipe in pipes_small.values():
                pipe.fit(X_small, y_small)
            models[dev]["small"] = pipes_small
            print(f"  -> Small Regime trained on {len(dev_small)} samples.")
        else:
            print("  -> Skipped Small Regime (Not enough data)")

        if len(dev_large) >= 3:
            pipes_large = get_base_pipelines()
            X_large = dev_large[features]
            y_large = dev_large["total_iops"]
            for pipe in pipes_large.values():
                pipe.fit(X_large, y_large)
            models[dev]["large"] = pipes_large
            print(f"  -> Large Regime trained on {len(dev_large)} samples.")
        else:
            print("  -> Skipped Large Regime (Not enough data)")

    return models


# ------------------------------------------------------------------ #
# Regime-based prediction
# ------------------------------------------------------------------ #
def evaluate_and_predict(df_test, models, features):
    """Route prediction through the correct regime model."""
    print("\n--- Starting Regime-Based Prediction on Test Data ---")
    all_results = []

    for dev, dev_regimes in models.items():
        dev_test = df_test[df_test["device"] == dev].copy()
        if dev_test.empty:
            continue

        dev_results = dev_test[["device", "read_percent", "block_size_kb"]].copy()
        if "actual_total_iops" in dev_test.columns:
            dev_results["Actual"] = dev_test["actual_total_iops"]
        else:
            dev_results["Actual"] = dev_test["total_iops"]

        for m in [m for m in MODEL_ORDER if m != "Actual"]:
            dev_results[m] = np.nan

        mask_small = dev_test["block_size_kb"] <= REGIME_THRESHOLD
        mask_large = dev_test["block_size_kb"] > REGIME_THRESHOLD

        if mask_small.any() and dev_regimes["small"]:
            X_small = dev_test[mask_small][features]
            for name, model in dev_regimes["small"].items():
                preds = model.predict(X_small)
                preds[preds < 0] = 0
                dev_results.loc[mask_small, name] = preds

        if mask_large.any() and dev_regimes["large"]:
            X_large = dev_test[mask_large][features]
            for name, model in dev_regimes["large"].items():
                preds = model.predict(X_large)
                preds[preds < 0] = 0
                dev_results.loc[mask_large, name] = preds

        all_results.append(dev_results)

    if not all_results:
        return None
    return pd.concat(all_results, ignore_index=True)


# ------------------------------------------------------------------ #
# Visualization – 2-panel grid plots
# ------------------------------------------------------------------ #
def create_plots(df, error_line_color="navy", error_line_width=2.5,
                 error_line_alpha=0.9):
    """Two-panel plot: error percentage on top, IOPS bars on bottom."""
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
        ax_err.set_ylabel("    Error (%)", fontsize=FONT_SIZE_LABEL)
        ax_err.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
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
                    if pd.isna(p):
                        errors.append(np.nan)
                    else:
                        errors.append((p - a) / a * 100 if a > 0 else 0)

            ax_err.plot(x[wi] + offsets, errors, marker="o",
                        linewidth=error_line_width, alpha=error_line_alpha,
                        color=error_line_color)

        ax_err.grid(axis="y", linestyle="--", alpha=0.4)

        # Bar panel
        for i, m in enumerate(MODEL_ORDER):
            ax_bar.bar(x + offsets[i], g[m].fillna(0).values / 1000,
                       width=bar_w, color=BAR_COLORS[m],
                       hatch=HATCH_MAP[m], edgecolor="black")

        ax_bar.set_xlabel("Read %", fontsize=FONT_SIZE_LABEL)
        ax_bar.set_ylabel("IOPS (x1000)", fontsize=FONT_SIZE_LABEL)
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(reads, fontsize=FONT_SIZE_TICK)
        ax_bar.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)

        legend_handles = [
            mpatches.Patch(facecolor=BAR_COLORS[m], hatch=HATCH_MAP[m],
                           edgecolor="black", label=LEGEND_LABELS[m])
            for m in MODEL_ORDER
        ]
        fig.legend(handles=legend_handles, loc="upper center",
                   bbox_to_anchor=(0.53, 1.06), ncol=5,
                   fontsize=FONT_SIZE_LEGEND, frameon=True,
                   handlelength=1.4, handletextpad=0.4,
                   columnspacing=0.8, labelspacing=0.3, borderpad=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        if EXPORT_PDF:
            fname = f"{PLOT_DIR}/bar_error_grid_{dev}_{bs}k.{PLOT_EXT}"
            plt.savefig(fname, bbox_inches="tight")
        plt.close()


# ------------------------------------------------------------------ #
# Summary MAPE plot
# ------------------------------------------------------------------ #
def create_summary_plot(results_df, bar_width=0.24):
    """Grouped bar chart of MAPE per device."""
    print("\n--- Generating Final Summary Plot ---")

    model_cols = [m for m in MODEL_ORDER if m != "Actual"]
    device_labels = {"nvme0n": "NVMe", "nvme1n": "Optane", "pmem": "Pmem"}

    summary = []
    for m in model_cols:
        tmp = results_df[results_df["Actual"] > 0].copy()
        tmp = tmp.dropna(subset=[m])
        if tmp.empty:
            continue
        tmp["APE"] = np.abs((tmp[m] - tmp["Actual"]) / tmp["Actual"]) * 100
        s = tmp.groupby("device")["APE"].mean().reset_index()
        s["Model"] = m
        summary.append(s)

    if not summary:
        return
    plot_df = pd.concat(summary, ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    devices = plot_df["device"].unique()
    x = np.arange(len(devices))
    n_models = len(model_cols)
    offsets = (np.arange(n_models) - (n_models - 1) / 2) * bar_width

    for i, m in enumerate(model_cols):
        vals_s = plot_df[plot_df["Model"] == m].set_index("device")["APE"]
        vals = [vals_s.get(d, 0) for d in devices]

        bars = ax.bar(x + offsets[i], vals, width=bar_width,
                      color=BAR_COLORS[m], hatch=HATCH_MAP[m],
                      edgecolor="black")
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2,
                        h + ax.get_ylim()[1] * 0.015,
                        f"{h:.0f}%", ha="center", va="bottom",
                        fontsize=FONT_SIZE_TICK - 4)

    ax.set_xticks(x)
    ax.set_xticklabels([device_labels.get(d, d) for d in devices],
                       fontsize=FONT_SIZE_TICK)
    ax.set_ylabel("Mean Absolute \nPercentage Error (%)",
                  fontsize=FONT_SIZE_LABEL)
    ax.tick_params(axis="y", labelsize=FONT_SIZE_TICK)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend_handles = [
        mpatches.Patch(facecolor=BAR_COLORS[m], hatch=HATCH_MAP[m],
                       edgecolor="black", label=LEGEND_LABELS[m])
        for m in model_cols
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.28), ncol=4,
              fontsize=FONT_SIZE_LEGEND, handlelength=1.3,
              handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.3, borderpad=0.3)

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
        description="Block-size generalization with regime-based training.",
    )
    parser.add_argument("data_path",
                        help="Path to unified data file (e.g. AllDevices2.xlsx)")
    args = parser.parse_args()

    # 1. Load data
    df_raw = read_and_prep_data(args.data_path, is_test_file=False)
    if df_raw is None:
        return

    # 2. Feature engineering – exclude sata/sda devices
    df_eng = engineer_features(df_raw)
    print("Filtering out 'sda' devices...")
    df_eng = df_eng[df_eng["device"].str.lower() != "sda"]

    # 3. Split: train on powers-of-2, test on non-standard block sizes
    print("Splitting Data based on Block Size (Train=Powers of 2, Test=Others)...")
    train_df = df_eng[df_eng["block_size_kb"].isin(STANDARD_BLOCK_SIZES)].copy()
    test_df = df_eng[~df_eng["block_size_kb"].isin(STANDARD_BLOCK_SIZES)].copy()

    print(f"  - Train Data Shape: {train_df.shape}")
    print(f"  - Test Data Shape: {test_df.shape}")

    if test_df.empty:
        print("CRITICAL: No testing data found. Exiting.")
        return

    test_df["actual_total_iops"] = test_df["total_iops"]

    # 4. Train
    models = train_models(train_df, FEATURES)

    # 5. Predict
    res = evaluate_and_predict(test_df, models, FEATURES)

    if res is not None:
        agg = res.groupby(
            ["device", "block_size_kb", "read_percent"], as_index=False,
        ).mean(numeric_only=True)

        agg.to_excel("predictions_regime_based.xlsx", index=False)
        create_plots(agg)
        create_summary_plot(agg)
        print("\nScript Completed Successfully!")


if __name__ == "__main__":
    main()
