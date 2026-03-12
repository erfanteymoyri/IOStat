"""
Motivational Utilization Curves
================================
Plots True Utilization vs. IOstat-Reported Utilization as smooth curves
for each storage device found in the data file.  An error arrow highlights
the discrepancy at a configurable data point.

Each device produces a separate plot page inside a single PDF output.

Usage:
    python 02_motivational_curves.py <path_to_excel_file>

Output: motivational_utilization_curves.pdf
"""

import argparse
import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from utils import smooth_curve

# ---------------------------------------------------------------
# Style
# ---------------------------------------------------------------
sns.set_theme(style="whitegrid")

TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 18
LEGEND_FONTSIZE = 18
ERROR_TEXT_FONTSIZE = 18

# Index of the data point where the error arrow is drawn
ERROR_POINT_INDEX = 4


def main(data_path, output_pdf="motivational_utilization_curves.pdf"):
    """Generate per-device utilization curve plots."""
    df = pd.read_excel(data_path, sheet_name=0)

    devices = df["job options/filename"].unique()
    pdf = PdfPages(output_pdf)

    for device in devices:
        plt.figure(figsize=(5, 3))
        ddf = df[df["job options/filename"] == device]

        for rw in ddf["job options/rw"].unique():
            wdf = ddf[ddf["job options/rw"] == rw].sort_values("limitation")

            fio_iops = (wdf["read/iops"] + wdf["write/iops"]) / 1000
            true_util = wdf["limitation"]

            iostat_iops = (wdf["avg_rps"] + wdf["avg_wps"]) / 1000
            iostat_util = wdf["avg_util"]

            # Smooth curves
            sx1, sy1 = smooth_curve(fio_iops.values, true_util.values)
            sx2, sy2 = smooth_curve(iostat_iops.values, iostat_util.values)

            plt.plot(sx1, sy1, label="True", linewidth=2.5)
            plt.plot(sx2, sy2, label="Iostat", linewidth=2.5)

            # Error arrow at the selected data point
            if ERROR_POINT_INDEX < len(fio_iops):
                x1 = float(fio_iops.iloc[ERROR_POINT_INDEX])
                y1 = float(true_util.iloc[ERROR_POINT_INDEX])
                x2 = float(iostat_iops.iloc[ERROR_POINT_INDEX])
                y2 = float(iostat_util.iloc[ERROR_POINT_INDEX])
                error_value = y2 - y1

                plt.annotate(
                    "",
                    xy=(x2, y2),
                    xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=3, color="red"),
                )
                plt.text(
                    (x1 + x2) / 2,
                    (y1 + y2) / 2 + 2,
                    f" Error={error_value:.1f}%",
                    color="red",
                    fontsize=ERROR_TEXT_FONTSIZE,
                    ha="left",
                )

        # Labels, ticks, legend
        plt.xlabel("IOPS (x1000)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("Utilization (%)", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks(fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.legend(fontsize=LEGEND_FONTSIZE)
        plt.tight_layout()

        pdf.savefig()
        plt.show()

    pdf.close()
    print(f"PDF saved as: {output_pdf}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot motivational utilization curves per device."
    )
    parser.add_argument("data_path", help="Path to the Excel data file.")
    parser.add_argument(
        "--output", default="motivational_utilization_curves.pdf",
        help="Output PDF file name (default: motivational_utilization_curves.pdf).",
    )
    args = parser.parse_args()
    main(args.data_path, args.output)
