"""
Smooth-Curve Evaluation (IOPS vs Utilization)
==============================================
Plots smooth utilization curves for True, IOstat, and Ours (model-based)
for each device.  Error arrows annotate the discrepancy at a selected
data point.

Requires a pre-trained model dictionary (from 13_execute_training.py or
loaded via joblib).

Usage:
    python 15_evaluation_smooth_curves.py <eval_data> <models_joblib>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_smooth_curve_evaluation


def main(data_path, models_path, target_model="LASSO_POLY", output=None):
    """Load models and run the smooth-curve evaluation plot."""
    models = joblib.load(models_path)
    print(f"Loaded models from: {models_path}")

    plot_smooth_curve_evaluation(
        data_path,
        models,
        target_model=target_model,
        output_pdf=output,
        error_point_index=2,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smooth-curve evaluation plot with error arrows.",
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default=None, help="Output PDF path.")
    args = parser.parse_args()
    main(args.data_path, args.models_path, args.model, args.output)
