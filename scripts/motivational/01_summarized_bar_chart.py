"""
Summarized Motivational Bar Chart
==================================
Generates a horizontal bar chart comparing True Utilization vs.
IOstat-Reported Utilization across different IOPS levels.

Single in-flight I/O requests (low IOPS) and multiple in-flight I/O
requests (high IOPS) are shown in separate background-shaded regions
to highlight that IOstat's error grows dramatically under concurrent
workloads.

Output: summarized_motiv.pdf
"""

import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (
    MOTIVATIONAL_IOPS_SINGLE,
    MOTIVATIONAL_TRUE_SINGLE,
    MOTIVATIONAL_IOSTAT_SINGLE,
    MOTIVATIONAL_IOPS_MULTI,
    MOTIVATIONAL_TRUE_MULTI,
    MOTIVATIONAL_IOSTAT_MULTI,
)

# ---------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------
X_LABEL_FONTSIZE = 22
Y_LABEL_FONTSIZE = 22
X_TICK_FONTSIZE = 20
Y_TICK_FONTSIZE = 20
LEGEND_FONTSIZE = 18

# ---------------------------------------------------------------
# Legend positions
# ---------------------------------------------------------------
LEGEND1_LOC = "lower center"
LEGEND1_BBOX = (0.7, 0.12)

LEGEND2_LOC = "upper right"
LEGEND2_BBOX = (1.0, 0.73)

# ---------------------------------------------------------------
# Background colours for the two regions
# ---------------------------------------------------------------
SINGLE_BG_COLOR = "#b2f1cd"
MULTI_BG_COLOR = "#fff9c4"

# ---------------------------------------------------------------
# Axis range
# ---------------------------------------------------------------
X_MIN = 0
X_MAX = 100
X_TICKS = [0, 20, 40, 60, 80, 100]

# ---------------------------------------------------------------
# Combine data
# ---------------------------------------------------------------
labels = MOTIVATIONAL_IOPS_SINGLE + MOTIVATIONAL_IOPS_MULTI
true_vals = MOTIVATIONAL_TRUE_SINGLE + MOTIVATIONAL_TRUE_MULTI
iostat_vals = MOTIVATIONAL_IOSTAT_SINGLE + MOTIVATIONAL_IOSTAT_MULTI

n_single = len(MOTIVATIONAL_IOPS_SINGLE)
n_total = len(labels)

# ---------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
plt.rcParams.update({"font.size": 20})

index = np.arange(n_total)
bar_height = 0.35

# Bar colours
true_color = "#3C6EB4"
iostat_color = "#ffb7d6"

# ---------------------------------------------------------------
# Background shading
# ---------------------------------------------------------------
ax.axhspan(-0.5, n_single - 0.5,
           facecolor=SINGLE_BG_COLOR, alpha=0.55, zorder=0)
ax.axhspan(n_single - 0.5, n_total - 0.5,
           facecolor=MULTI_BG_COLOR, alpha=0.55, zorder=0)

# Dotted separator line between single and multi regions
separator_y = n_single - 0.5
ax.axhline(separator_y, color="gray", linestyle="--",
           linewidth=2, alpha=0.9, zorder=10)

# ---------------------------------------------------------------
# Draw bars
# ---------------------------------------------------------------
ax.barh(index - bar_height / 2, true_vals, height=bar_height,
        label="True Utilization", color=true_color, zorder=2)

ax.barh(index + bar_height / 2, iostat_vals, height=bar_height,
        label="IOstat-Reported Utilization",
        color=iostat_color, edgecolor="black", hatch="///", zorder=2)

# ---------------------------------------------------------------
# Legend 1: True vs IOstat
# ---------------------------------------------------------------
leg1 = ax.legend(
    ncol=1, loc=LEGEND1_LOC,
    bbox_to_anchor=LEGEND1_BBOX,
    fontsize=LEGEND_FONTSIZE,
    frameon=False,
)

# ---------------------------------------------------------------
# Legend 2: Single vs Multi background bands
# ---------------------------------------------------------------
single_patch = Patch(facecolor=SINGLE_BG_COLOR, edgecolor="none",
                     label="Single In-Flight I/O Requests")
multi_patch = Patch(facecolor=MULTI_BG_COLOR, edgecolor="none",
                    label="Multiple In-Flight I/O Requests")

leg2 = ax.legend(
    ncol=1, handles=[multi_patch, single_patch],
    loc=LEGEND2_LOC,
    bbox_to_anchor=LEGEND2_BBOX,
    fontsize=LEGEND_FONTSIZE,
    frameon=False,
)
ax.add_artist(leg1)

# ---------------------------------------------------------------
# Axes and labels
# ---------------------------------------------------------------
ax.set_xlabel("Utilization (%)", fontsize=X_LABEL_FONTSIZE)
ax.set_ylabel("IOPS (x1000)", fontsize=Y_LABEL_FONTSIZE)

ax.set_xticks(X_TICKS)
ax.set_xticklabels(X_TICKS, fontsize=X_TICK_FONTSIZE)

ax.set_yticks(index)
ax.set_yticklabels(labels, fontsize=Y_TICK_FONTSIZE)

ax.set_xlim(X_MIN, X_MAX)
ax.grid(axis="x", linestyle="--", alpha=0.95, zorder=5)

plt.tight_layout()

# ---------------------------------------------------------------
# Save output
# ---------------------------------------------------------------
fig.savefig("summarized_motiv.pdf", format="pdf", bbox_inches="tight")
plt.show()
