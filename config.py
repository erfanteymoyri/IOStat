"""
Shared configuration for the IOStat project.

Centralizes constants, hyperparameters, and visual style definitions used
across notebooks and scripts for predicting I/O device utilization with
ML models.
"""

# ---------------------------------------------------------------------------
# Plot font sizes — standard preset
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE = 16
FONT_SIZE_LABEL = 14
FONT_SIZE_TICK = 12
FONT_SIZE_LEGEND = 12

# ---------------------------------------------------------------------------
# Plot font sizes — large preset (evaluation plots)
# ---------------------------------------------------------------------------
FONT_SIZE_LABEL_LARGE = 24
FONT_SIZE_TICK_LARGE = 20
FONT_SIZE_LEGEND_LARGE = 20
FONT_SIZE_ERROR_TEXT = 18

# ---------------------------------------------------------------------------
# Plot font sizes — mentor style preset
# ---------------------------------------------------------------------------
FONT_SIZE_LABEL_MENTOR = 21
FONT_SIZE_TICK_MENTOR = 21
FONT_SIZE_LEGEND_MENTOR = 20

# ---------------------------------------------------------------------------
# ML model hyperparameters
# ---------------------------------------------------------------------------
HYPERPARAMS = {
    "lasso_degree": 3,
    "lasso_alpha": 0.1,
    "rf_n_estimators": 100,
    "rf_max_depth": None,
    "rf_min_samples_leaf": 1,
}

# ---------------------------------------------------------------------------
# Model ordering and visual styles
# ---------------------------------------------------------------------------
MODEL_ORDER = ["Actual", "LINEAR", "LASSO_POLY", "RANDOM_FOREST", "SVR"]
MODEL_ORDER_WITH_HYBRID = [
    "Actual",
    "LINEAR",
    "LASSO_POLY",
    "RANDOM_FOREST",
    "SVR",
    "SMART_HYBRID",
]

HATCH_MAP = {
    "Actual": "///",
    "LINEAR": "xxx",
    "LASSO_POLY": "+++",
    "RANDOM_FOREST": "\\\\",
    "SVR": "---",
    "SMART_HYBRID": "**",
}

BAR_COLORS = {
    "Actual": "#d896ff",
    "LINEAR": "#ffcb85",
    "LASSO_POLY": "#c2ffb4",
    "RANDOM_FOREST": "#99ccff",
    "SVR": "#ffd4e5",
    "SMART_HYBRID": "#ffd700",
}

LEGEND_LABELS = {
    "Actual": "Real",
    "LINEAR": "Linear",
    "LASSO_POLY": "Lasso Poly",
    "RANDOM_FOREST": "Random Forest",
    "SVR": "SVR",
    "SMART_HYBRID": "Smart Hybrid",
}

# ---------------------------------------------------------------------------
# Error panel configuration
# ---------------------------------------------------------------------------
ERROR_PANEL_RATIO = 1.0
EXPORT_PDF = True

# ---------------------------------------------------------------------------
# Evaluation line-plot colors
# ---------------------------------------------------------------------------
IOSTAT_LINE_COLOR = "red"
TRUE_UTIL_LINE_COLOR = "green"
OURS_LINE_COLOR = "blue"
NEW_MODEL_LINE_COLOR = "darkorange"

ERROR_COLOR_IOSTAT = "red"
ERROR_COLOR_OURS = "darkgreen"

# ---------------------------------------------------------------------------
# Features used by ML models
# ---------------------------------------------------------------------------
FEATURES = ["read_percent", "block_size_kb"]

# ---------------------------------------------------------------------------
# Block-size regime splitting and standard sizes
# ---------------------------------------------------------------------------
BS_THRESHOLD = 32
STANDARD_BLOCK_SIZES = [2, 4, 8, 16, 32, 64, 128]
ANCHOR_BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# ---------------------------------------------------------------------------
# Hatches for block-size bar charts
# ---------------------------------------------------------------------------
BLOCK_SIZE_HATCHES = ["///", "xxx", "+++"]

# ---------------------------------------------------------------------------
# Motivational plot data (single-stream and multi-stream)
# ---------------------------------------------------------------------------
MOTIVATIONAL_IOPS_SINGLE = ["5", "10", "20"]
MOTIVATIONAL_TRUE_SINGLE = [9, 18, 35]
MOTIVATIONAL_IOSTAT_SINGLE = [9, 15, 30]

MOTIVATIONAL_IOPS_MULTI = ["100", "200"]
MOTIVATIONAL_TRUE_MULTI = [18, 35]
MOTIVATIONAL_IOSTAT_MULTI = [80, 93]
