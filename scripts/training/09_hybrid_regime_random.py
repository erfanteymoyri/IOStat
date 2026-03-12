"""
Hybrid approach: regime-based training with stratified random split.

Combines regime splitting (BS_THRESHOLD = 32 KB) with random train/test
partitioning.  Runs two scenarios: 60/40 and 70/30.  Excludes SATA/SDA
devices.

Usage
-----
    python 09_hybrid_regime_random.py <data_path>
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
    FONT_SIZE_TITLE, FONT_SIZE_LABEL, FONT_SIZE_TICK, FONT_SIZE_LEGEND,
    HYPERPARAMS, MODEL_ORDER, HATCH_MAP, BAR_COLORS, LEGEND_LABELS,
    ERROR_PANEL_RATIO, EXPORT_PDF, FEATURES,
)
from utils import read_and_prep_data, engineer_features, get_base_pipelines

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
BS_THRESHOLD = 32
SEED = 42
PLOT_EXT = "pdf"


# ------------------------------------------------------------------ #
# Regime-based training
# ------------------------------------------------------------------ #
def train_models(df_train, features):
    """Train separate models for small and large BS regimes."""
    print("\n--- Starting Regime-Based Model Training ---")
    models = {}

    for dev in df_train["device"].unique():
        if dev == "unknown":
            continue

        dev_data = df_train[df_train["device"] == dev].copy()
        dev_small = dev_data[dev_data["block_size_kb"] <= BS_THRESHOLD]
        # Gap of 1000 KB above threshold to separate regimes (original design)
        dev_large = dev_data[dev_data["block_size_kb"] > BS_THRESHOLD + 1000]

        models[dev] = {"small": {}, "large": {}}
        print(f"Training models for device: {dev}")

        if len(dev_small) >= 3:
            pipes = get_base_pipelines()
            X = dev_small[features]
            y = dev_small["total_iops"]
            for pipe in pipes.values():
                pipe.fit(X, y)
            models[dev]["small"] = pipes
            print(f"  -> Small Regime (<= {BS_THRESHOLD}k): {len(dev_small)} samples.")
        else:
            print("  -> Skipped Small Regime (Not enough data)")

        if len(dev_large) >= 3:
            pipes = get_base_pipelines()
            X = dev_large[features]
            y = dev_large["total_iops"]
            for pipe in pipes.values():
                pipe.fit(X, y)
            models[dev]["large"] = pipes
            print(f"  -> Large Regime (> {BS_THRESHOLD}k): {len(dev_large)} samples.")
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

        mask_small = dev_test["block_size_kb"] <= BS_THRESHOLD
        # Same gap as training to ensure consistent regime boundaries
        mask_large = dev_test["block_size_kb"] > BS_THRESHOLD + 1000

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
def create_plots(df, plot_dir, error_line_color="navy",
                 error_line_width=2.5, error_line_alpha=0.9):
    """Two-panel plot: error on top, IOPS bars on bottom."""
    print(f"\n--- Generating Detailed Grid Plots in: {plot_dir} ---")
    os.makedirs(plot_dir, exist_ok=True)

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
            fname = f"{plot_dir}/bar_error_grid_{dev}_{bs}k.{PLOT_EXT}"
            plt.savefig(fname, bbox_inches="tight")
        plt.close()


# ------------------------------------------------------------------ #
# Summary MAPE plot
# ------------------------------------------------------------------ #
def create_summary_plot(results_df, plot_dir, title_suffix="",
                        bar_width=0.24):
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

    plt.title(f"Final Summary: Average MAPE by Device\n{title_suffix}",
              fontsize=FONT_SIZE_TITLE)

    legend_handles = [
        mpatches.Patch(facecolor=BAR_COLORS[m], hatch=HATCH_MAP[m],
                       edgecolor="black", label=LEGEND_LABELS[m])
        for m in model_cols
    ]
    ax.legend(handles=legend_handles, loc="upper center",
              bbox_to_anchor=(0.5, 1.30), ncol=4,
              fontsize=FONT_SIZE_LEGEND, handlelength=1.3,
              handletextpad=0.4, columnspacing=0.8,
              labelspacing=0.3, borderpad=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fname = f"{plot_dir}/Overall_Error.{PLOT_EXT}"
    if EXPORT_PDF:
        plt.savefig(fname, bbox_inches="tight")
    plt.close()
    print(f"Summary plot saved to {fname}")


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="Hybrid regime-based + stratified random split.",
    )
    parser.add_argument("data_path",
                        help="Path to unified data file (e.g. AllDevices2.xlsx)")
    args = parser.parse_args()

    # 1. Load and engineer features
    df_raw = read_and_prep_data(args.data_path, is_test_file=False)
    if df_raw is None:
        return

    df_eng = engineer_features(df_raw)
    print("Filtering out 'sda' devices...")
    df_eng = df_eng[df_eng["device"].str.lower() != "sda"]

    # 2. Run experiments
    split_ratios = [0.6, 0.7]

    for train_frac in split_ratios:
        test_frac = 1.0 - train_frac
        ratio_name = f"{int(train_frac * 100)}_{int(test_frac * 100)}"

        print("\n" + "=" * 60)
        print(f" RUNNING EXPERIMENT: {int(train_frac * 100)}% Train / "
              f"{int(test_frac * 100)}% Test")
        print(f" Using Regime Split Threshold: BS = {BS_THRESHOLD}k")
        print("=" * 60)

        plot_dir = f"barcharts_regime_random_{ratio_name}"

        # Stratified random split per device AND per regime
        train_list, test_list = [], []
        for dev in df_eng["device"].unique():
            dev_data = df_eng[df_eng["device"] == dev]

            dev_small = dev_data[dev_data["block_size_kb"] <= BS_THRESHOLD]
            dev_large = dev_data[dev_data["block_size_kb"] > BS_THRESHOLD]

            if not dev_small.empty:
                train_s = dev_small.sample(frac=train_frac, random_state=SEED)
                test_s = dev_small.drop(train_s.index)
                train_list.append(train_s)
                test_list.append(test_s)

            if not dev_large.empty:
                train_l = dev_large.sample(frac=train_frac, random_state=SEED)
                test_l = dev_large.drop(train_l.index)
                train_list.append(train_l)
                test_list.append(test_l)

        train_df = pd.concat(train_list)
        test_df = pd.concat(test_list)

        print(f"  - Total Train Data Shape: {train_df.shape}")
        print(f"  - Total Test Data Shape: {test_df.shape}")

        if test_df.empty:
            print("CRITICAL: No testing data found. Skipping.")
            continue

        test_df["actual_total_iops"] = test_df["total_iops"]

        # Train (regime-based)
        models = train_models(train_df, FEATURES)

        # Predict (regime-based)
        res = evaluate_and_predict(test_df, models, FEATURES)

        if res is not None:
            agg = res.groupby(
                ["device", "block_size_kb", "read_percent"], as_index=False,
            ).mean(numeric_only=True)

            excel_name = f"predictions_regime_random_{ratio_name}.xlsx"
            agg.to_excel(excel_name, index=False)
            create_plots(agg, plot_dir)

            title_text = (f"Regime Split (BS Threshold={BS_THRESHOLD}k) | "
                          f"{int(train_frac * 100)}% Train / "
                          f"{int(test_frac * 100)}% Test")
            create_summary_plot(agg, plot_dir, title_suffix=title_text)
            print(f"{ratio_name} Split Completed Successfully!")

    print("\nALL EXPERIMENTS FINISHED!")


if __name__ == "__main__":
    main()
