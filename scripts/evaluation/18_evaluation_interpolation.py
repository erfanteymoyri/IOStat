"""
Evaluation with Interpolated Prediction
=========================================
Uses linear interpolation between standard anchor block sizes to predict
IOPS for non-standard block sizes, then computes utilization and plots
time-series comparisons.

This addresses the limitation where direct model prediction on unusual
block sizes (e.g. 23KB) can be inaccurate because the model was trained
only on standard sizes (4, 8, 16, 32, ...).

Usage:
    python 18_evaluation_interpolation.py <eval_data> <models_joblib>
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
)
from utils import feature_engineering_evaluation, predict_with_interpolation

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Font sizes
AXIS_LABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 20
TICK_FONTSIZE = 20


def main(data_path, models_path, target_model="LASSO_POLY", output=None):
    """Evaluate with interpolated prediction and plot time-series."""
    models = joblib.load(models_path)
    print(f"Loaded models from: {models_path}")

    df = pd.read_excel(data_path, sheet_name=0)
    df = feature_engineering_evaluation(df)

    if output is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        output = f"{base}_interpolated.pdf"

    pdf = PdfPages(output)
    legend_bbox = (0.65, 0.89)

    for dev_path in df["job options/filename"].unique():
        plt.figure(figsize=(6, 4))
        ddf = df[df["job options/filename"] == dev_path].copy()
        if ddf.empty:
            plt.close()
            continue

        dev_name = ddf["device"].iloc[0]
        dev_models = models.get(dev_name, {})
        model = dev_models.get(target_model, None)
        if model is None:
            plt.close()
            continue

        time_axis = (
            ddf["time"] if "time" in ddf.columns else np.arange(len(ddf))
        )
        fio_iops = ddf["read/iops"].fillna(0) + ddf["write/iops"].fillna(0)
        true_util = ddf["limitation"]
        iostat_util = ddf["avg_util"]

        # Interpolated prediction
        X = ddf[FEATURES].values
        max_iops_pred = predict_with_interpolation(model, X)
        max_iops_pred = np.maximum(max_iops_pred, 1.0)
        mask = max_iops_pred > 0
        ours_util = np.zeros(len(fio_iops))
        ours_util[mask] = (fio_iops.values[mask] / max_iops_pred[mask]) * 100

        # Plot
        plt.plot(
            time_axis, iostat_util,
            label="IOstat Util", color=IOSTAT_LINE_COLOR, linewidth=2,
        )
        plt.plot(
            time_axis, true_util,
            label="Avg. True Util", color=TRUE_UTIL_LINE_COLOR,
            linestyle="--", linewidth=2,
        )
        plt.plot(
            time_axis[mask], ours_util[mask],
            label="Ours Util (Interpolated)", color=OURS_LINE_COLOR,
            linewidth=2,
        )

        plt.xlabel("Time (x10 seconds)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Utilization (%)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, max(110, max(iostat_util.max(), ours_util.max()) + 10))
        plt.legend(fontsize=LEGEND_FONTSIZE, loc="upper right",
                   bbox_to_anchor=legend_bbox)
        plt.tight_layout()
        pdf.savefig()
        plt.show()

    pdf.close()
    print(f"PDF saved as: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation with interpolated prediction.",
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default=None, help="Output PDF path.")
    args = parser.parse_args()
    main(args.data_path, args.models_path, args.model, args.output)
