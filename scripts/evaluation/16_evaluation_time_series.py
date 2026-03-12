"""
Time-Series Utilization Evaluation
====================================
Plots utilization over time for each device: IOstat, True Utilization,
and Ours (predicted from a trained model).

This is the standard evaluation plot used for NVMe and other device
workloads such as A667_QD1, A17, etc.

Usage:
    python 16_evaluation_time_series.py <eval_data> <models_joblib> [--model LASSO_POLY]
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_time_series_evaluation


def main(data_path, models_path, target_model="LASSO_POLY", output=None):
    """Load models and produce a time-series utilization plot."""
    models = joblib.load(models_path)
    print(f"Loaded models from: {models_path}")

    plot_time_series_evaluation(
        data_path,
        models,
        target_model=target_model,
        output_pdf=output,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-series utilization evaluation plot.",
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default=None, help="Output PDF path.")
    args = parser.parse_args()
    main(args.data_path, args.models_path, args.model, args.output)
