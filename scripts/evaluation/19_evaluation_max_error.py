"""
Evaluation with Maximum-Error Arrow
=====================================
Plots smooth utilization curves (True, IOstat, Ours) and automatically
places error arrows at the point of maximum discrepancy rather than at a
fixed index.

Usage:
    python 19_evaluation_max_error.py <eval_data> <models_joblib>
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES, ERROR_COLOR_IOSTAT, ERROR_COLOR_OURS
from utils import (
    feature_engineering_evaluation,
    smooth_curve,
    max_error_index,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# Font sizes
AXIS_LABEL_FONTSIZE = 22
TICK_FONTSIZE = 22
LEGEND_FONTSIZE = 20
ERROR_TEXT_FONTSIZE = 21


def main(data_path, models_path, target_model="LASSO_POLY", output=None):
    """Produce smooth-curve evaluation plots with maximum-error arrows."""
    models = joblib.load(models_path)
    print(f"Loaded models from: {models_path}")

    df = pd.read_excel(data_path, sheet_name=0)
    df = feature_engineering_evaluation(df)

    if output is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        output = f"{base}_max_error.pdf"

    devices = df["job options/filename"].unique()
    pdf = PdfPages(output)

    for dev_path in devices:
        plt.figure(figsize=(6, 4))
        ddf = df[df["job options/filename"] == dev_path].sort_values("limitation")
        dev_name = ddf["device"].iloc[0]

        dev_models = models.get(dev_name, {})
        model = dev_models.get(target_model, None)
        if model is None:
            plt.close()
            continue

        fio_iops = (ddf["read/iops"] + ddf["write/iops"]) / 1000
        true_util = ddf["limitation"].values
        iostat_iops = (ddf["avg_rps"] + ddf["avg_wps"]) / 1000
        iostat_util = ddf["avg_util"].values

        # Ours utilization
        X = ddf[FEATURES].values
        max_iops_pred = model.predict(X)
        mask = max_iops_pred > 0
        ours_util = (fio_iops.values[mask] * 1000 / max_iops_pred[mask]) * 100

        # Smooth curves
        sx_t, sy_t = smooth_curve(fio_iops.values, true_util)
        sx_i, sy_i = smooth_curve(iostat_iops.values, iostat_util)
        sx_o, sy_o = smooth_curve(fio_iops.values[mask], ours_util)

        plt.plot(sx_t, sy_t, label="True", linewidth=2.5)
        plt.plot(sx_i, sy_i, label="Iostat", linewidth=2.5)
        plt.plot(sx_o, sy_o, label="Ours", linestyle="--", linewidth=2.5)

        # Find the point of maximum error for IOstat
        idx = max_error_index(true_util, iostat_util)
        if idx < len(fio_iops):
            x_true = float(fio_iops.iloc[idx])
            y_true = float(true_util[idx])
            x_io = float(iostat_iops.iloc[idx])
            y_io = float(iostat_util[idx])
            err_io = y_io - y_true

            plt.annotate(
                "", xy=(x_io, y_io), xytext=(x_true, y_true),
                arrowprops=dict(arrowstyle="->", lw=3, color=ERROR_COLOR_IOSTAT),
            )
            plt.text(
                (x_true + x_io) / 2, (y_true + y_io) / 2 + 2,
                f" IOstat Err={err_io:.1f}%",
                color=ERROR_COLOR_IOSTAT, fontsize=ERROR_TEXT_FONTSIZE, ha="left",
            )

        # Find the point of maximum error for Ours
        common_len = min(len(true_util[mask]), len(ours_util))
        if common_len > 0:
            idx_o = max_error_index(true_util[mask][:common_len], ours_util[:common_len])
            y_true_o = float(true_util[mask][idx_o])
            y_ours = float(ours_util[idx_o])
            x_pos = float(fio_iops.values[mask][idx_o])
            err_ours = y_ours - y_true_o

            plt.annotate(
                "", xy=(x_pos, y_ours), xytext=(x_pos, y_true_o),
                arrowprops=dict(arrowstyle="->", lw=3, color=ERROR_COLOR_OURS),
            )
            plt.text(
                x_pos + 0.5, (y_true_o + y_ours) / 2,
                f" Ours Err={err_ours:.1f}%",
                color=ERROR_COLOR_OURS, fontsize=ERROR_TEXT_FONTSIZE, ha="left",
            )

        plt.xlabel("IOPS (x1000)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Utilization (%)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()
        pdf.savefig()
        plt.show()

    pdf.close()
    print(f"PDF saved as: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluation with max-error arrow annotation.",
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default=None, help="Output PDF path.")
    args = parser.parse_args()
    main(args.data_path, args.models_path, args.model, args.output)
