"""
Evaluation — FIUMail J4 QD2 — Basic Time-Series
=================================================
Plots IOstat, True Utilization, and model-predicted utilization for the
FIUMail workload replay (Job 4, Queue Depth 2).

Usage:
    python 24_eval_fiumail_qd2_basic.py <eval_data> <models_joblib>
"""

import argparse
import os
import sys

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_time_series_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FIUMail J4 QD2 basic evaluation."
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default="evaluation_fiumail_j4_qd2.pdf")
    args = parser.parse_args()

    models = joblib.load(args.models_path)
    plot_time_series_evaluation(
        args.data_path, models,
        target_model=args.model,
        output_pdf=args.output,
    )
