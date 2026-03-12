"""
Evaluation — Workload A42 (QD1) — Model Comparison
====================================================
Compares IOstat, True, Old Model (baseline), and New 60/40 RF Model
for workload A42 QD1.  Includes hardcoded baseline predictions.

Usage:
    python 21_eval_workload_42_comparison.py <eval_data> <old_models> <new_models>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_comparison_evaluation

# Hardcoded baseline predictions for workload A42
HARDCODED_PREDICTIONS_A42 = [
    30.71919714, 63.41900943, 63.46192839,
    67.7510765, 63.66108931, 63.75432706,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Workload A42 QD1 model comparison."
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("old_models", help="Path to old models (.joblib).")
    parser.add_argument("new_models", help="Path to 60/40 models (.joblib).")
    parser.add_argument("--output", default="evaluation_workload_42_QD1_comparison.pdf")
    args = parser.parse_args()

    old_models = joblib.load(args.old_models)
    plot_comparison_evaluation(
        args.data_path,
        old_models=old_models,
        new_models_path=args.new_models,
        old_model_name="LASSO_POLY",
        new_model_name="RANDOM_FOREST",
        hardcoded_predictions=HARDCODED_PREDICTIONS_A42,
        output_pdf=args.output,
    )
