"""
Shared evaluation and plotting utilities.

Provides reusable functions for time-series utilization plots, smooth-curve
evaluation plots, and model comparison visualizations used across all
workload evaluation scripts.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MaxNLocator

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import (
    IOSTAT_LINE_COLOR,
    TRUE_UTIL_LINE_COLOR,
    OURS_LINE_COLOR,
    NEW_MODEL_LINE_COLOR,
    FEATURES,
)
from utils import feature_engineering_evaluation, smooth_curve

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


def plot_time_series_evaluation(
    data_path,
    models,
    target_model="LASSO_POLY",
    output_pdf=None,
    axis_label_fontsize=24,
    legend_fontsize=20,
    tick_fontsize=20,
    figsize=(6, 4),
    legend_bbox=(0.65, 0.89),
    divide_iops_by=1,
):
    """
    Plot time-series utilization comparison: IOstat vs True vs Ours.

    Parameters
    ----------
    data_path : str
        Path to the evaluation Excel file.
    models : dict
        Mapping of device name -> dict of model name -> fitted model.
    target_model : str
        Which model to use for "Ours" predictions (e.g. 'LASSO_POLY').
    output_pdf : str or None
        If given, save plots into this PDF.
    divide_iops_by : int
        Divisor for IOPS display (use 1000 for x1000 scale, 1 for raw).
    """
    df = pd.read_excel(data_path, sheet_name=0)
    df = feature_engineering_evaluation(df)

    if output_pdf is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        output_pdf = f"{base}.pdf"

    pdf = PdfPages(output_pdf)

    for dev_path in df["job options/filename"].dropna().unique():
        plt.figure(figsize=figsize)
        ddf = df[df["job options/filename"] == dev_path].copy()
        dev_name = ddf["device"].iloc[0]

        dev_models = models.get(dev_name, {})
        chosen_model = dev_models.get(target_model, None)
        if chosen_model is None:
            plt.close()
            continue

        time_axis = (
            ddf["time"] if "time" in ddf.columns else np.arange(len(ddf))
        )
        fio_iops = ddf["read/iops"].fillna(0) + ddf["write/iops"].fillna(0)
        true_util = ddf["limitation"]
        iostat_util = ddf["avg_util"]

        # Prediction
        X = ddf[FEATURES].values
        max_iops_pred = chosen_model.predict(X)
        max_iops_pred = np.maximum(max_iops_pred, 1.0)
        mask = max_iops_pred > 0
        ours_util = np.zeros(len(fio_iops))
        ours_util[mask] = (
            fio_iops.values[mask] / (max_iops_pred[mask] * divide_iops_by)
        ) * 100

        # Plot lines
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
            label="Ours Util", color=OURS_LINE_COLOR, linewidth=2,
        )

        plt.xlabel("Time (x10 seconds)", fontsize=axis_label_fontsize)
        plt.ylabel("Utilization (%)", fontsize=axis_label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, max(110, max(iostat_util.max(), ours_util.max()) + 10))
        plt.legend(
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=legend_bbox,
        )
        plt.tight_layout()
        pdf.savefig()
        plt.show()

    pdf.close()
    print(f"PDF saved as: {output_pdf}")


def plot_comparison_evaluation(
    data_path,
    old_models,
    new_models_path=None,
    old_model_name="LASSO_POLY",
    new_model_name="RANDOM_FOREST",
    hardcoded_predictions=None,
    output_pdf=None,
    axis_label_fontsize=24,
    legend_fontsize=16,
    tick_fontsize=23,
    figsize=(5, 4),
    export_excel=True,
):
    """
    Plot comparison of IOstat, True, Old Model, New 60/40 Model, and
    optionally hardcoded predictions.

    Parameters
    ----------
    data_path : str
        Path to the evaluation Excel file.
    old_models : dict
        Mapping device -> {model_name: model} from the original training.
    new_models_path : str or None
        Path to a joblib file with 60/40 trained models.
    old_model_name : str
        Model key in old_models to use.
    new_model_name : str
        Model key in new_models to use.
    hardcoded_predictions : list or None
        Pre-computed prediction values to overlay.
    output_pdf : str or None
        Output PDF path.
    export_excel : bool
        Whether to also export results as an Excel file.
    """
    import joblib

    df = pd.read_excel(data_path, sheet_name=0)
    df = feature_engineering_evaluation(df)

    if output_pdf is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        output_pdf = f"{base}.pdf"

    # Load new models if path provided
    new_models = {}
    if new_models_path is not None:
        try:
            new_models = joblib.load(new_models_path)
            print(f"Loaded new models from: {new_models_path}")
        except Exception as e:
            print(f"WARNING: Could not load models from {new_models_path}: {e}")

    pdf = PdfPages(output_pdf)
    all_predictions = []

    if hardcoded_predictions is not None:
        ours_util_hardcoded = np.array(hardcoded_predictions)

    for dev_path in df["job options/filename"].dropna().unique():
        plt.figure(figsize=figsize)
        ddf = df[df["job options/filename"] == dev_path].copy()
        dev_name = ddf["device"].iloc[0]

        time_axis = (
            ddf["time"] if "time" in ddf.columns else np.arange(len(ddf))
        )
        fio_iops = ddf["read/iops"].fillna(0) + ddf["write/iops"].fillna(0)
        true_util = ddf["limitation"]
        iostat_util = ddf["avg_util"]
        X = ddf[FEATURES].values

        # Old model prediction
        max_iops_pred_old = np.zeros(len(fio_iops))
        ours_util_old = np.zeros(len(fio_iops))
        try:
            dev_models_old = old_models.get(dev_name, {})
            old_model = dev_models_old.get(old_model_name, None)
        except Exception:
            old_model = None

        if old_model is not None:
            max_iops_pred_old = old_model.predict(X)
            mask_old = max_iops_pred_old > 0
            ours_util_old[mask_old] = (
                fio_iops.values[mask_old] / max_iops_pred_old[mask_old]
            ) * 100

        # New 60/40 model prediction
        max_iops_pred_new = np.zeros(len(fio_iops))
        ours_util_new = np.zeros(len(fio_iops))
        dev_models_new = new_models.get(dev_name, {})
        new_model = dev_models_new.get(new_model_name, None)

        if new_model is not None:
            max_iops_pred_new = new_model.predict(X)
            mask_new = max_iops_pred_new > 0
            ours_util_new[mask_new] = (
                fio_iops.values[mask_new] / max_iops_pred_new[mask_new]
            ) * 100

        # Plot
        plt.plot(
            time_axis, iostat_util,
            label="IOstat", color=IOSTAT_LINE_COLOR, linewidth=2.5,
        )
        plt.plot(
            time_axis, true_util,
            label="True Util", color=TRUE_UTIL_LINE_COLOR,
            linestyle="--", linewidth=2.5,
        )

        if hardcoded_predictions is not None and len(ours_util_hardcoded) == len(time_axis):
            plt.plot(
                time_axis, ours_util_hardcoded,
                label="Ours (Baseline)", color=OURS_LINE_COLOR, linewidth=2.5,
            )

        if new_model is not None:
            mask_new = max_iops_pred_new > 0
            plt.plot(
                time_axis[mask_new], ours_util_new[mask_new],
                label=f"Ours ({new_model_name})",
                color=NEW_MODEL_LINE_COLOR, linewidth=2.5,
            )

        plt.xlabel("Time (x10 seconds)", fontsize=axis_label_fontsize)
        plt.ylabel("Utilization (%)", fontsize=axis_label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.ylim(0, max(110, max(iostat_util.max(), ours_util_new.max()) + 10))
        plt.legend(
            fontsize=legend_fontsize,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
        )
        plt.tight_layout()
        pdf.savefig()
        plt.show()

        # Collect predictions for export
        pred_df = pd.DataFrame({
            "Device": ddf["device"],
            "Block Size (KB)": ddf["block_size_kb"],
            "Read Percent": ddf["read_percent"],
            "Time": time_axis,
            "True IOPS": fio_iops,
            "True Utilization (%)": true_util,
            "IOstat Utilization (%)": iostat_util,
            "Old Model Predicted Max IOPS": max_iops_pred_old,
            "Old Model Utilization (%)": ours_util_old,
            "New Model Predicted Max IOPS": max_iops_pred_new,
            "New Model Utilization (%)": ours_util_new,
        })
        all_predictions.append(pred_df)

    pdf.close()
    print(f"PDF saved as: {output_pdf}")

    if export_excel and all_predictions:
        excel_path = output_pdf.replace(".pdf", "_results.xlsx")
        pd.concat(all_predictions, ignore_index=True).to_excel(
            excel_path, index=False
        )
        print(f"Excel saved as: {excel_path}")


def plot_smooth_curve_evaluation(
    data_path,
    models,
    target_model="LASSO_POLY",
    output_pdf=None,
    error_point_index=2,
    axis_label_fontsize=24,
    legend_fontsize=20,
    tick_fontsize=20,
    error_text_fontsize=18,
    figsize=(6, 4),
):
    """
    Plot smooth utilization curves with error arrows for IOstat vs True vs Ours.

    Parameters
    ----------
    data_path : str
        Path to the evaluation Excel file.
    models : dict
        Mapping device -> {model_name: model}.
    target_model : str
        Model to use for "Ours" utilization curve.
    error_point_index : int
        Index of the data point for the error arrow annotation.
    """
    df = pd.read_excel(data_path, sheet_name=0)
    df = feature_engineering_evaluation(df)

    if output_pdf is None:
        base = os.path.splitext(os.path.basename(data_path))[0]
        output_pdf = f"{base}_smooth.pdf"

    devices = df["job options/filename"].unique()
    pdf = PdfPages(output_pdf)

    for dev_path in devices:
        plt.figure(figsize=figsize)
        ddf = df[df["job options/filename"] == dev_path].sort_values("limitation")
        dev_name = ddf["device"].iloc[0]

        dev_models = models.get(dev_name, {})
        model = dev_models.get(target_model, None)
        if model is None:
            plt.close()
            continue

        fio_iops = (ddf["read/iops"] + ddf["write/iops"]) / 1000
        true_util = ddf["limitation"]
        iostat_iops = (ddf["avg_rps"] + ddf["avg_wps"]) / 1000
        iostat_util = ddf["avg_util"]

        # Prediction
        X = ddf[FEATURES].values
        max_iops_pred = model.predict(X)
        mask = max_iops_pred > 0
        ours_util = (fio_iops.values[mask] * 1000 / max_iops_pred[mask]) * 100

        # Smooth curves
        sx_t, sy_t = smooth_curve(fio_iops.values, true_util.values)
        sx_i, sy_i = smooth_curve(iostat_iops.values, iostat_util.values)
        sx_o, sy_o = smooth_curve(fio_iops.values[mask], ours_util)

        plt.plot(sx_t, sy_t, label="True", linewidth=2.8)
        plt.plot(sx_i, sy_i, label="Iostat", linewidth=2.6)
        plt.plot(sx_o, sy_o, label="Ours", linestyle="--", linewidth=2.6)

        # Error arrows
        if error_point_index < len(fio_iops) and error_point_index < len(ours_util):
            x_true = float(fio_iops.iloc[error_point_index])
            y_true = float(true_util.iloc[error_point_index])
            x_io = float(iostat_iops.iloc[error_point_index])
            y_io = float(iostat_util.iloc[error_point_index])
            y_ours = float(ours_util[error_point_index])

            err_io = y_io - y_true
            err_ours = y_ours - y_true

            # IOstat error arrow
            plt.annotate(
                "", xy=(x_io, y_io), xytext=(x_true, y_true),
                arrowprops=dict(arrowstyle="->", lw=3, color="red"),
            )
            plt.text(
                (x_true + x_io) / 2, (y_true + y_io) / 2 + 2,
                f" IOstat Err={err_io:.1f}%",
                color="red", fontsize=error_text_fontsize, ha="left",
            )

            # Ours error arrow
            plt.annotate(
                "", xy=(x_true, y_ours), xytext=(x_true, y_true),
                arrowprops=dict(arrowstyle="->", lw=3, color="darkgreen"),
            )
            plt.text(
                x_true + 0.5, (y_true + y_ours) / 2,
                f" Ours Err={err_ours:.1f}%",
                color="darkgreen", fontsize=error_text_fontsize, ha="left",
            )

        plt.xlabel("IOPS (x1000)", fontsize=axis_label_fontsize)
        plt.ylabel("Utilization (%)", fontsize=axis_label_fontsize)
        plt.xticks(fontsize=tick_fontsize)
        plt.yticks(fontsize=tick_fontsize)
        plt.legend(fontsize=legend_fontsize)
        plt.tight_layout()
        pdf.savefig()
        plt.show()

    pdf.close()
    print(f"PDF saved as: {output_pdf}")
