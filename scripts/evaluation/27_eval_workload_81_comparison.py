"""
Evaluation — Workload A81 (QD1) — Model Comparison
====================================================
Compares models for workload A81 QD1.

Usage:
    python 27_eval_workload_81_comparison.py <eval_data> <old_models> <new_models>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_comparison_evaluation

HARDCODED_PREDICTIONS_A81 = [
    12.64742543, 51.40074737, 55.66380173, 51.00023179, 55.89563851, 54.25023136,
    66.38452631, 50.28716433, 53.00123693, 50.93659847, 53.29174671, 53.36882073,
    51.69084043, 51.62533555, 49.76975387, 51.79003354, 51.55608752, 49.13290821,
    51.57480321, 49.48592293, 49.07614203, 49.28901523, 49.65976938, 50.91788279,
    50.10325521, 51.44192187, 47.84103602, 49.05840259, 49.15951736, 49.56042855,
    51.35395817, 15.00762493,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workload A81 QD1 model comparison.")
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("old_models", help="Path to old models (.joblib).")
    parser.add_argument("new_models", help="Path to 60/40 models (.joblib).")
    parser.add_argument("--output", default="evaluation_workload_81_QD1_comparison.pdf")
    args = parser.parse_args()

    old_models = joblib.load(args.old_models)
    plot_comparison_evaluation(
        args.data_path, old_models=old_models, new_models_path=args.new_models,
        old_model_name="LASSO_POLY", new_model_name="RANDOM_FOREST",
        hardcoded_predictions=HARDCODED_PREDICTIONS_A81, output_pdf=args.output,
    )
