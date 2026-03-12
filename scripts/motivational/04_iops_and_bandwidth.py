"""
IOPS and Bandwidth Grouped Bar Chart
=====================================
Creates a dual-axis grouped bar chart for a single storage device:
  - Left Y-axis : Max IOPS (x1000) per workload and block size (bars).
  - Right Y-axis: Bandwidth in GB (line overlay).

Hatched patterns distinguish block sizes (2k, 4k, 8k, etc.).

Usage:
    python 04_iops_and_bandwidth.py <path_to_excel_file> --device <device_name>

Output: iops_and_bandwidth.pdf
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import load_data, smooth_curve
from config import BLOCK_SIZE_HATCHES

# ---------------------------------------------------------------
# Hatches per block size
# ---------------------------------------------------------------
HATCHES = BLOCK_SIZE_HATCHES


def plot_iops_and_bandwidth(
    path,
    device,
    workloads=None,
    block_sizes=None,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xticks_fontsize=10,
    yticks_fontsize=10,
    xticks_label=None,
    legend_loc="upper right",
    legend_bbox_to_anchor=(1.05, 1),
    save_pdf=False,
    file_name="plot.pdf",
    bw_ymax=3.0,
    bw_divisor=1_000_000,
):
    """
    Plot IOPS bars and bandwidth lines for a specific device.

    Parameters
    ----------
    path : str
        Path to the performance results file.
    device : str
        Device name to filter on.
    workloads : list or None
        Workloads to include (None = all).
    block_sizes : list or None
        Block sizes to include (None = all).
    save_pdf : bool
        Whether to save output as PDF.
    file_name : str
        Output PDF name.
    """
    df = load_data(path)
    df_dev = df[df["device"] == device]
    if df_dev.empty:
        raise ValueError(f"No data for device={device}")

    if workloads is not None:
        df_dev = df_dev[df_dev["workload"].isin(workloads)]
    if block_sizes is not None:
        df_dev = df_dev[df_dev["block_size"].isin(block_sizes)]

    # IOPS pivot table
    iops_table = (
        df_dev.groupby(["workload", "block_size"])["iops"]
        .max()
        .unstack("block_size")
        .sort_index()
    )
    workloads_order = iops_table.index.tolist()
    block_sizes_order = iops_table.columns.tolist()

    # Bandwidth pivot table
    bw_table = (
        df_dev.groupby(["workload", "block_size"])["bandwidth_kbps"]
        .max()
        .unstack("block_size")
        .reindex(index=workloads_order, columns=block_sizes_order)
    )

    x = np.arange(len(workloads_order))
    bar_width = 0.18

    # Figure and IOPS bars
    fig, ax = plt.subplots(figsize=(7, 4))

    base_colors = ["#80ff80", "#8080ff", "#ffc0ff"]
    colors = [base_colors[i % len(base_colors)] for i in range(len(block_sizes_order))]

    for i, (bs, c) in enumerate(zip(block_sizes_order, colors)):
        ax.bar(
            x + i * bar_width,
            iops_table[bs].values / 1000,
            width=bar_width,
            color=c,
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )

    ax.set_ylabel("Max IOPS (x1000)", fontsize=ylabel_fontsize)
    ax.set_xlabel("Workload", fontsize=xlabel_fontsize)

    if xticks_label is None:
        xticks_label = workloads_order

    ax.set_xticks(x + bar_width * (len(block_sizes_order) - 1) / 2)
    ax.set_xticklabels(xticks_label, fontsize=xticks_fontsize)
    ax.tick_params(axis="y", labelsize=yticks_fontsize)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    # Custom legend with hatches
    legend_handles = [
        mpatches.Patch(
            facecolor=colors[i],
            hatch=HATCHES[i % len(HATCHES)],
            edgecolor="black",
            label=bs,
        )
        for i, bs in enumerate(block_sizes_order)
    ]

    legend = ax.legend(
        handles=legend_handles,
        fontsize=17,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
        ncol=3,
        frameon=False,
    )

    # "Block Sizes" label beside legend
    legend_box = legend.get_window_extent(fig.canvas.get_renderer())
    inv = ax.transAxes.inverted()
    legend_bbox_axes = inv.transform(legend_box)

    ax.text(
        legend_bbox_axes[0][0] - 0.04,
        (legend_bbox_axes[0][1] + legend_bbox_axes[1][1]) / 2,
        "Block Sizes",
        fontsize=18,
        va="top",
        ha="right",
        transform=ax.transAxes,
    )

    # Bandwidth lines on the right y-axis
    ax2 = ax.twinx()
    ax2.set_ylabel("Bandwidth (GB)", fontsize=ylabel_fontsize)
    ax2.tick_params(axis="y", labelsize=yticks_fontsize)
    ax2.set_ylim(0, bw_ymax)

    ax2.set_zorder(ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    for i, (bs, c) in enumerate(zip(block_sizes_order, colors)):
        bw_vals = bw_table[bs].values / bw_divisor
        sx, sy = smooth_curve(x + i * bar_width, bw_vals)
        ax2.plot(sx, sy, color=c, linewidth=2.5, zorder=10)

    plt.tight_layout()

    if save_pdf:
        plt.savefig(file_name, format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grouped bar chart of IOPS and bandwidth per workload."
    )
    parser.add_argument("data_path", help="Path to the performance results file.")
    parser.add_argument("--device", required=True, help="Device to filter on.")
    parser.add_argument("--output", default="iops_and_bandwidth.pdf")
    args = parser.parse_args()

    plot_iops_and_bandwidth(
        args.data_path,
        device=args.device,
        xlabel_fontsize=20,
        ylabel_fontsize=20,
        xticks_fontsize=20,
        yticks_fontsize=20,
        legend_loc="upper right",
        legend_bbox_to_anchor=(1.05, 1),
        save_pdf=True,
        file_name=args.output,
    )
