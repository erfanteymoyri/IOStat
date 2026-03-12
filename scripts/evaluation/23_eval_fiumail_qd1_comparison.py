"""
Evaluation — FIUMail J4 QD1 — Model Comparison
================================================
Compares IOstat, True, Old Model (baseline), and New 60/40 RF Model
for the FIUMail workload (Job 4, QD1).

Usage:
    python 23_eval_fiumail_qd1_comparison.py <eval_data> <old_models> <new_models>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_comparison_evaluation

# Hardcoded baseline predictions for FIUMail J4 QD1
HARDCODED_PREDICTIONS_FIUMAIL_QD1 = [
    29.6597401, 27.260759, 25.76569309, 26.15138134,
    25.65238884, 25.09509153, 26.11392243, 25.52659379,
    24.98490527, 25.60838949, 27.06001574, 25.4131784,
    25.43815745, 24.94844968, 26.11379441, 26.28037608,
    20.94651447,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIUMail J4 QD1 model comparison."
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("old_models", help="Path to old models (.joblib).")
    parser.add_argument("new_models", help="Path to 60/40 models (.joblib).")
    parser.add_argument("--output", default="evaluation_fiumail_j4_qd1_comparison.pdf")
    args = parser.parse_args()

    old_models = joblib.load(args.old_models)
    plot_comparison_evaluation(
        args.data_path,
        old_models=old_models,
        new_models_path=args.new_models,
        old_model_name="LASSO_POLY",
        new_model_name="RANDOM_FOREST",
        hardcoded_predictions=HARDCODED_PREDICTIONS_FIUMAIL_QD1,
        output_pdf=args.output,
    )
