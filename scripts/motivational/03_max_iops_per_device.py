"""
Max IOPS Per Device Bar Chart
==============================
Plots the maximum observed IOPS for each storage device in a
publication-quality bar chart with rounded corners and soft drop shadows.

The chart demonstrates the wide performance gap between different
device types (PMEM, Optane, NVMe, SATA SSD).

Usage:
    python 03_max_iops_per_device.py <path_to_excel_file>

Output: challenge_max_IOPS_per_dev.pdf
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import load_data


def plot_max_iops_per_device(
    path,
    workload,
    block_size,
    engine=None,
    poll=None,
    title_fontsize=16,
    xlabel_fontsize=12,
    ylabel_fontsize=12,
    xticks_fontsize=10,
    yticks_fontsize=10,
    xticks_label=None,
    xticks_label_rotation=20,
    save_pdf=False,
    file_name="plot.pdf",
    show_values=True,
    bar_width=0.25,
):
    """
    Plot the maximum IOPS per device for a given workload and block size.

    Parameters
    ----------
    path : str
        Path to the performance results file.
    workload : str
        Workload filter (e.g. 'randread').
    block_size : str
        Block size filter (e.g. '2k').
    engine : str or None
        I/O engine filter (e.g. 'libaio').
    poll : str or None
        Polling mode filter.
    save_pdf : bool
        Whether to save the plot as a PDF.
    file_name : str
        Output PDF file name.
    """
    df = load_data(path)

    # Filter dataset
    df_sel = df[(df["workload"] == workload) & (df["block_size"] == block_size)]
    if engine is not None:
        df_sel = df_sel[df_sel["engine"] == engine]
    if poll is not None:
        df_sel = df_sel[df_sel["poll"] == poll]

    if df_sel.empty:
        raise ValueError(
            f"No rows found for workload={workload}, block_size={block_size}, "
            f"engine={engine}, poll={poll}"
        )

    # Extract max IOPS per device
    grouped = (
        df_sel.groupby("device", as_index=False)["iops"]
        .max()
        .rename(columns={"iops": "max_iops"})
        .sort_values("max_iops", ascending=False)
    )
    devices = grouped["device"].tolist()
    max_iops = grouped["max_iops"].values / 1000  # convert to K IOPS

    x = np.arange(len(devices))

    # Figure style
    plt.figure(figsize=(8, 3.5))
    ax = plt.gca()

    base_color = "#d8b4ff"
    edge_color = "#1A4E89"

    # Draw bars with rounded corners and soft shadow
    for xi, val in zip(x, max_iops):
        # Shadow
        ax.bar(xi, val, width=bar_width, color="black", alpha=0.08, zorder=2)

        # Main rounded bar
        bar = FancyBboxPatch(
            (xi - bar_width / 2, 0),
            bar_width,
            val,
            boxstyle="round,pad=0.15,rounding_size=0.15",
            linewidth=1.0,
            edgecolor=edge_color,
            facecolor=base_color,
            zorder=5,
        )
        ax.add_patch(bar)

    # Value labels above bars
    if show_values:
        for xi, val in zip(x, max_iops):
            ax.text(
                xi,
                val + max(max_iops) * 0.03,
                f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=xticks_fontsize,
                color="black",
                fontweight="bold",
            )

    # Labels and axes
    ax.set_ylabel("Max IOPS (x1000)", fontsize=ylabel_fontsize)
    ax.set_xlabel("Device", fontsize=xlabel_fontsize)

    if xticks_label is None:
        xticks_label = devices

    ax.set_xticks(x)
    ax.set_xticklabels(
        xticks_label, rotation=xticks_label_rotation, fontsize=xticks_fontsize
    )
    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticks(), fontsize=yticks_fontsize)

    ax.grid(axis="y", linestyle="--", alpha=0.45, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_pdf:
        plt.savefig(file_name, format="pdf", bbox_inches="tight")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bar chart of max IOPS per storage device."
    )
    parser.add_argument("data_path", help="Path to the performance results file.")
    parser.add_argument("--workload", default="randread", help="Workload type.")
    parser.add_argument("--block_size", default="2k", help="Block size.")
    parser.add_argument("--engine", default="libaio", help="I/O engine.")
    parser.add_argument("--poll", default="none", help="Polling mode.")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["PMEM", "Optane", "NVMe", "Sata SSD"],
        help="Custom x-axis labels.",
    )
    parser.add_argument("--output", default="challenge_max_IOPS_per_dev.pdf")
    args = parser.parse_args()

    plot_max_iops_per_device(
        args.data_path,
        workload=args.workload,
        block_size=args.block_size,
        engine=args.engine,
        poll=args.poll,
        xlabel_fontsize=20,
        ylabel_fontsize=20,
        xticks_fontsize=19,
        yticks_fontsize=19,
        xticks_label=args.labels,
        xticks_label_rotation=0,
        save_pdf=True,
        file_name=args.output,
        show_values=True,
        bar_width=0.25,
    )
