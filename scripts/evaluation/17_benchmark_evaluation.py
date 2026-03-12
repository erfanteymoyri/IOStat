"""
Benchmark Evaluation with Saved Models
========================================
Loads pre-trained models from a joblib file and evaluates them against
a new benchmark workload.  Produces time-series utilization plots and
an optional CSV/Excel export of all predictions.

Usage:
    python 17_benchmark_evaluation.py <benchmark_data> <models_joblib> [--model RANDOM_FOREST]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (
    FEATURES,
    IOSTAT_LINE_COLOR,
    TRUE_UTIL_LINE_COLOR,
    OURS_LINE_COLOR,
    FONT_SIZE_LABEL_LARGE,
    FONT_SIZE_TICK_LARGE,
    FONT_SIZE_LEGEND_LARGE,
)
from utils import feature_engineering_evaluation

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def process_benchmark(benchmark_file, models, target_model="RANDOM_FOREST"):
    """
    Evaluate a trained model against a benchmark workload file.

    Parameters
    ----------
    benchmark_file : str
        Path to the evaluation Excel file.
    models : dict
        Mapping of device name -> {model_name: fitted_model}.
    target_model : str
        Which model to use for prediction.

    Returns
    -------
    pd.DataFrame or None
        Combined predictions across all devices.
    """
    df = pd.read_excel(benchmark_file, sheet_name=0)
    df = feature_engineering_evaluation(df)

    pdf_name = os.path.splitext(benchmark_file)[0] + f"_{target_model}_plot.pdf"
    pdf = PdfPages(pdf_name)
    all_predictions = []

    for dev_path in df["job options/filename"].unique():
        plt.figure(figsize=(12, 6))
        ddf = df[df["job options/filename"] == dev_path].copy()
        dev_name = ddf["device"].iloc[0]

        dev_models = models.get(dev_name, {})
        chosen_model = dev_models.get(target_model, None)

        if chosen_model is None:
            print(f"Warning: No trained '{target_model}' found for device '{dev_name}'. Skipping.")
            plt.close()
            continue

        time_axis = ddf["time"] if "time" in ddf.columns else np.arange(len(ddf))
        fio_iops = ddf["read/iops"].fillna(0) + ddf["write/iops"].fillna(0)
        true_util = ddf["limitation"]
        iostat_util = ddf["avg_util"]

        # Prediction
        X = ddf[FEATURES].values
        max_iops_pred = chosen_model.predict(X)
        mask = max_iops_pred > 0
        ours_util = np.zeros(len(fio_iops))
        ours_util[mask] = (fio_iops.values[mask] / max_iops_pred[mask]) * 100

        # Plot
        plt.plot(time_axis, iostat_util, label="IOstat Util",
                 color=IOSTAT_LINE_COLOR, linewidth=2.5)
        plt.plot(time_axis, true_util, label="Avg. True Util",
                 color=TRUE_UTIL_LINE_COLOR, linestyle="--", linewidth=2.5)
        plt.plot(time_axis[mask], ours_util[mask],
                 label=f"Ours Util ({target_model})",
                 color=OURS_LINE_COLOR, linewidth=2.5)

        plt.xlabel("Time (x10 seconds)", fontsize=FONT_SIZE_LABEL_LARGE)
        plt.ylabel("Utilization (%)", fontsize=FONT_SIZE_LABEL_LARGE)
        plt.xticks(fontsize=FONT_SIZE_TICK_LARGE)
        plt.yticks(fontsize=FONT_SIZE_TICK_LARGE)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, max(110, max(iostat_util.max(), ours_util.max()) + 10))
        plt.legend(
            fontsize=FONT_SIZE_LEGEND_LARGE,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
        )
        plt.tight_layout()
        pdf.savefig()
        plt.show()

        # Collect predictions
        pred_df = pd.DataFrame({
            "Device": ddf["device"],
            "Block Size (KB)": ddf["block_size_kb"],
            "Read Percent": ddf["read_percent"],
            "Time": time_axis,
            "True IOPS": fio_iops,
            "True Utilization (%)": true_util,
            f"Predicted Max IOPS ({target_model})": max_iops_pred,
            "Predicted Utilization (%)": ours_util,
        })
        all_predictions.append(pred_df)

    pdf.close()
    print(f"PDF saved as: {pdf_name}")

    if all_predictions:
        combined = pd.concat(all_predictions, ignore_index=True)
        csv_name = os.path.splitext(benchmark_file)[0] + f"_{target_model}_predictions.csv"
        combined.to_csv(csv_name, index=False)
        print(f"CSV saved as: {csv_name}")
        return combined
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate saved models against a benchmark workload.",
    )
    parser.add_argument("benchmark_data", help="Path to benchmark Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="RANDOM_FOREST", help="Model to use.")
    args = parser.parse_args()

    loaded_models = joblib.load(args.models_path)
    print(f"Loaded models from: {args.models_path}")
    process_benchmark(args.benchmark_data, loaded_models, args.model)
