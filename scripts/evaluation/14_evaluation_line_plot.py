"""
Simple Evaluation Line Plot
============================
Loads evaluation results and draws IOstat vs. Ours utilization curves
on the same axes for a quick visual comparison.

Usage:
    python 14_evaluation_line_plot.py <evaluation_data_path>
"""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import IOSTAT_LINE_COLOR, OURS_LINE_COLOR

sns.set_theme(style="whitegrid")


def main(data_path):
    """Plot IOstat vs Ours utilization from pre-computed evaluation data."""
    df = pd.read_excel(data_path, sheet_name="Sheet1")

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["TRUE"], df["IOstat"],
        label="IOstat Utilization", color=IOSTAT_LINE_COLOR, linewidth=2,
    )
    plt.plot(
        df["TRUE"], df["Ours"],
        label="Ours Utilization", color=OURS_LINE_COLOR, linewidth=2,
    )

    plt.xlabel("True", fontsize=14)
    plt.ylabel("Utilization (%)", fontsize=14)
    plt.title("Comparison of IOstat vs. Ours Utilization", fontsize=16)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple IOstat vs Ours utilization line plot.",
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    args = parser.parse_args()
    main(args.data_path)
