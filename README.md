# IOStat — Predicting Real Storage Device Utilization with Machine Learning

<p align="center">
  <strong>Python · scikit-learn · matplotlib</strong>
</p>

---

## Table of Contents

- [About the Project](#about-the-project)
- [Problem We Solve](#problem-we-solve)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Machine Learning Models](#machine-learning-models)
- [Usage](#usage)
  - [1. Training Models](#1-training-models)
  - [2. Evaluating Models](#2-evaluating-models)
  - [3. Motivational Charts](#3-motivational-charts)
- [Project Configuration](#project-configuration)
- [Model Input Features](#model-input-features)
- [Outputs](#outputs)
- [Dependencies](#dependencies)

---

## About the Project

**IOStat** is a Python-based machine learning project designed to **predict the maximum IOPS of storage devices** and accurately calculate **real utilization**.

This project uses **4 regression models** to predict the maximum achievable IOPS of a device given its workload parameters. After predicting the maximum IOPS, the real utilization is simply calculated as:

```
Real Utilization = (Current IOPS / Predicted Maximum IOPS) × 100
```

---

## Problem We Solve

The `iostat` tool in Linux calculates device utilization based on **a single concurrent I/O request**. This method reports **incorrect and much lower than actual values** when multiple I/O requests are executed concurrently.

| Scenario | IOPS | Real Utilization | iostat Report |
|----------|------|------------------|---------------|
| Single request — 5 IOPS | 5 | 9% | 9% ✅ |
| Single request — 20 IOPS | 20 | 35% | 30% ≈ |
| Multiple requests — 100 IOPS | 100 | 18% | 80% ❌ |
| Multiple requests — 200 IOPS | 200 | 35% | 93% ❌ |

> **Note:** The difference between single-request and multi-request scenarios is due to different device configurations and queue depth. In multi-request scenarios, the actual device capacity is much higher, but `iostat` does not account for this.

**Our Solution:** Using machine learning models, we predict the maximum IOPS of the device and accurately calculate the real utilization.

---

## Project Structure

```
IOStat/
├── config.py                    # Central configuration (hyperparameters, colors, fonts)
├── utils.py                     # Shared utilities (data loading, feature engineering, model building)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
└── scripts/
    ├── live_monitor.py              # Live iostat monitoring with ML predictions
    │
    ├── motivational/            # Motivational chart scripts
    │   ├── 01_summarized_bar_chart.py
    │   ├── 02_motivational_curves.py
    │   ├── 03_max_iops_per_device.py
    │   └── 04_iops_and_bandwidth.py
    │
    ├── training/                # Model training scripts
    │   ├── 05_deterministic_training.py
    │   ├── 06_blocksize_generalization.py
    │   ├── 07_smart_hybrid_model.py
    │   ├── 08_random_split_analysis.py
    │   ├── 09_hybrid_regime_random.py
    │   ├── 10_iops_bound_focused.py
    │   ├── 11_deterministic_mentor_viz.py
    │   └── 12_mentor_generalization_viz.py
    │
    └── evaluation/              # Evaluation and model comparison scripts
        ├── 13_execute_training.py
        ├── 14_evaluation_line_plot.py
        ├── 15_evaluation_smooth_curves.py
        ├── 16_evaluation_time_series.py
        ├── 17_benchmark_evaluation.py
        ├── 18_evaluation_interpolation.py
        ├── 19_evaluation_max_error.py
        ├── 20–30_eval_workload_*.py     # Various workload evaluations
        └── evaluation_utils.py          # Shared evaluation utilities
```

---

## Prerequisites

- **Python 3.8** or higher
- **pip** (Python package manager)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/erfanteymoyri/IOStat.git
cd IOStat
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

That's it! The project is ready to use. No build or compilation is required.

---

## Machine Learning Models

This project uses **4 regression models**, each implemented as a scikit-learn pipeline:

| Model | Description | Use Case |
|-------|-------------|----------|
| **LINEAR** | Simple linear regression with normalization | Baseline model for comparison |
| **LASSO_POLY** | Lasso regression with polynomial features (degree 3) | Capturing non-linear relationships |
| **RANDOM_FOREST** | Random Forest (100 trees) | Strong performance on diverse data |
| **SVR** | Support Vector Regression (RBF kernel) | Best performance for small block sizes |

In addition, there is a **SMART_HYBRID** model that routes between models based on block size:
- Block size ≤ 8 KB → **SVR**
- Block size 8 to 16 KB → **Random Forest**
- Block size > 16 KB → **Random Forest**

---

## Usage

> **Note:** All scripts should be run from the **project root directory** so that shared imports of `config.py` and `utils.py` work correctly.

### 1. Training Models

#### Train and Save Models

The first step is to train the models on training data and save them:

```bash
python scripts/evaluation/13_execute_training.py data/AllDevices.xlsx --save models.joblib
```

This script:
- Loads and preprocesses the training data
- Trains 4 models for each storage device
- Saves the models to the `models.joblib` file

#### Deterministic Training

Train on one dataset and test on another:

```bash
python scripts/training/05_deterministic_training.py data/AllDevices.xlsx data/limit100.xlsx
```

The output includes bar charts comparing actual vs. predicted IOPS and an Excel file with results.

#### Block Size Generalization

Train based on block size regime (small and large) and test on non-standard block sizes:

```bash
python scripts/training/06_blocksize_generalization.py data/AllDevices.xlsx
```

#### Smart Hybrid Model

Train the hybrid model that routes between SVR and Random Forest based on block size:

```bash
python scripts/training/07_smart_hybrid_model.py data/AllDevices.xlsx
```

#### Random Split Analysis

Analyze model performance with random data splitting:

```bash
python scripts/training/08_random_split_analysis.py data/AllDevices.xlsx
```

#### Other Training Scripts

| Script | Description |
|--------|-------------|
| `09_hybrid_regime_random.py` | Hybrid regime model with random splitting |
| `10_iops_bound_focused.py` | IOPS-bound focused training |
| `11_deterministic_mentor_viz.py` | Mentor-style visualization (large fonts) |
| `12_mentor_generalization_viz.py` | Mentor-style generalization analysis |

---

### 2. Evaluating Models

After training the models, you can evaluate their performance using the evaluation scripts:

#### Simple Line Plot

Quick comparison of IOstat utilization vs. our model:

```bash
python scripts/evaluation/14_evaluation_line_plot.py data/evaluation.xlsx
```

#### Smooth Curve Plot

Smoothed curves with error bars:

```bash
python scripts/evaluation/15_evaluation_smooth_curves.py data/evaluation.xlsx
```

#### Time Series Plot

Utilization comparison over time (IOstat vs. actual vs. our prediction):

```bash
python scripts/evaluation/16_evaluation_time_series.py data/evaluation.xlsx
```

#### Benchmark Evaluation

```bash
python scripts/evaluation/17_benchmark_evaluation.py
```

#### Interpolation Evaluation

Test predictions for non-standard block sizes using linear interpolation:

```bash
python scripts/evaluation/18_evaluation_interpolation.py data/evaluation.xlsx
```

#### Maximum Error Analysis

Find data points with the highest prediction error:

```bash
python scripts/evaluation/19_evaluation_max_error.py data/evaluation.xlsx
```

#### Specific Workload Evaluations

Each workload has two scripts: `basic` (baseline evaluation) and `comparison` (model comparison):

| Script | Workload |
|--------|----------|
| `20/21_eval_workload_42_*.py` | Workload 42 |
| `22/23_eval_fiumail_qd1_*.py` | FIUMail QD1 |
| `24/25_eval_fiumail_qd2_*.py` | FIUMail QD2 |
| `26/27_eval_workload_81_*.py` | Workload 81 |
| `28/29_eval_workload_669_*.py` | Workload 669 |
| `30_eval_workload_17_basic.py` | Workload 17 |

---

### 3. Live Monitoring (Real-Time Prediction)

The live monitor script uses `iostat` to read real-time device statistics and compares the standard `iostat` utilization report with our ML-predicted utilization. This makes the project **executable on any Linux system** with trained models.

#### Prerequisites

- `iostat` must be installed (`sudo apt install sysstat` on Debian/Ubuntu)
- Trained models saved as a `.joblib` file

#### Step 1: Train models on your device data

```bash
python scripts/evaluation/13_execute_training.py data/AllDevices.xlsx --save models.joblib
```

#### Step 2: Run the live monitor

```bash
# Basic usage (5-second interval, LASSO_POLY model)
python scripts/live_monitor.py models.joblib

# Custom interval and model
python scripts/live_monitor.py models.joblib --interval 3 --model RANDOM_FOREST

# Collect exactly 20 samples
python scripts/live_monitor.py models.joblib --count 20
```

The script will:
1. Run `iostat -txm -d` to discover available storage devices
2. Ask you to select which device to monitor
3. Automatically match the device to the trained model
4. Display a live comparison table showing IOstat utilization vs. our predicted utilization
5. Print summary statistics when you press `Ctrl+C`

**Example output:**

```
======================================================================
  Live Monitor | Device: sda | Model: LASSO_POLY | Interval: 5s
======================================================================
Sample   Time        IOPS  Read%   BS(kB)  IOstat%   Ours%     Diff
----------------------------------------------------------------------
1        14:23:05    150.0  66.7%    12.0     82.0%   21.3%   +60.7%
2        14:23:10    200.0  50.0%    16.0     93.0%   35.1%   +57.9%
3        14:23:15     80.0 100.0%     4.0     45.0%   12.8%   +32.2%
```

---

### 4. Motivational Charts

These scripts generate charts that **visually demonstrate the iostat problem**:

```bash
# Bar chart: Real utilization vs. iostat report
python scripts/motivational/01_summarized_bar_chart.py

# Motivational curves
python scripts/motivational/02_motivational_curves.py

# Maximum IOPS per device
python scripts/motivational/03_max_iops_per_device.py

# IOPS and bandwidth analysis
python scripts/motivational/04_iops_and_bandwidth.py
```

---

## Project Configuration

All central configuration is located in the `config.py` file:

### Hyperparameters

```python
HYPERPARAMS = {
    "lasso_degree": 3,          # Lasso polynomial degree
    "lasso_alpha": 0.1,         # Lasso regularization coefficient
    "rf_n_estimators": 100,     # Number of Random Forest trees
    "rf_max_depth": None,       # Maximum tree depth (unlimited)
    "rf_min_samples_leaf": 1,   # Minimum samples per leaf
}
```

### Standard and Anchor Block Sizes

```python
STANDARD_BLOCK_SIZES = [2, 4, 8, 16, 32, 64, 128]              # in KB
ANCHOR_BLOCK_SIZES = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
```

### Model Color Scheme

Each model has a specific color and pattern in charts:

| Model | Color | Pattern |
|-------|-------|---------|
| Actual | Purple `#d896ff` | `///` |
| LINEAR | Orange `#ffcb85` | `xxx` |
| LASSO_POLY | Green `#c2ffb4` | `+++` |
| RANDOM_FOREST | Blue `#99ccff` | `\\\\` |
| SVR | Pink `#ffd4e5` | `---` |
| SMART_HYBRID | Gold `#ffd700` | `**` |

---

## Model Input Features

The models use only **2 features**:

| Feature | Description | Range |
|---------|-------------|-------|
| `read_percent` | Percentage of read operations | 0 to 100 |
| `block_size_kb` | Block size in kilobytes | 2 to 4096 |

### Block Size Interpolation

For block sizes that are not directly present in the training data (e.g., 12 KB), the system automatically performs **linear interpolation** between the nearest anchor block sizes.

---

## Outputs

The scripts produce the following outputs:

| Type | Format | Description |
|------|--------|-------------|
| Charts | PDF | Comparison and evaluation charts |
| Results | Excel (`.xlsx`) | Numerical prediction result tables |
| Models | joblib (`.joblib`) | Saved trained models |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥ 1.21.0 | Numerical computations |
| pandas | ≥ 1.3.0 | Data management |
| matplotlib | ≥ 3.4.0 | Plotting |
| seaborn | ≥ 0.11.0 | Statistical charts |
| scipy | ≥ 1.7.0 | Curve interpolation |
| scikit-learn | ≥ 1.0.0 | Machine learning models |
| joblib | ≥ 1.1.0 | Model saving and loading |
| openpyxl | ≥ 3.0.0 | Reading and writing Excel files |

---

## Key Functions

### Data Loading and Preparation (`utils.py`)

```python
from utils import load_data, read_and_prep_data, engineer_features

# Simple loading of an Excel or CSV file
df = load_data("data/AllDevices.xlsx")

# Full loading and preprocessing for training
train_df = read_and_prep_data("data/AllDevices.xlsx", is_test_file=False)
train_df = engineer_features(train_df)

# Full loading and preprocessing for testing
test_df = read_and_prep_data("data/limit100.xlsx", is_test_file=True)
test_df = engineer_features(test_df)
```

### Building and Training Models

```python
from utils import get_base_pipelines
from config import FEATURES

# Build 4 model pipelines
pipelines = get_base_pipelines()

# Train each model on a single device's data
for name, pipeline in pipelines.items():
    pipeline.fit(X_train[FEATURES], y_train)
    predictions = pipeline.predict(X_test[FEATURES])
    print(f"{name}: {predictions[:5]}")
```

### Prediction with Interpolation

```python
from utils import predict_with_interpolation
import numpy as np

# Predict for non-standard block sizes
X_new = np.array([[50, 12], [100, 24]])  # [read_percent, block_size_kb]
predictions = predict_with_interpolation(trained_model, X_new)
```

---

## License

This project was developed for research and academic purposes.