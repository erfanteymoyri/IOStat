"""
Evaluation — FIUMail J4 QD2 — Model Comparison
================================================
Compares IOstat, True, Old Model, and New 60/40 RF Model for the FIUMail
workload (Job 4, QD2).

Usage:
    python 25_eval_fiumail_qd2_comparison.py <eval_data> <old_models> <new_models>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_comparison_evaluation

# Hardcoded baseline predictions for FIUMail J4 QD2
HARDCODED_PREDICTIONS_FIUMAIL_QD2 = [
    25.68757422, 70.44311094, 65.86707738,
    65.28734799, 66.38376974, 66.50011606, 68.41008626,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIUMail J4 QD2 model comparison."
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("old_models", help="Path to old models (.joblib).")
    parser.add_argument("new_models", help="Path to 60/40 models (.joblib).")
    parser.add_argument("--output", default="evaluation_fiumail_j4_qd2_comparison.pdf")
    args = parser.parse_args()

    old_models = joblib.load(args.old_models)
    plot_comparison_evaluation(
        args.data_path,
        old_models=old_models,
        new_models_path=args.new_models,
        old_model_name="LASSO_POLY",
        new_model_name="RANDOM_FOREST",
        hardcoded_predictions=HARDCODED_PREDICTIONS_FIUMAIL_QD2,
        output_pdf=args.output,
    )
