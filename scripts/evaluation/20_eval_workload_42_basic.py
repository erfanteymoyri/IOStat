"""
Evaluation — Workload A42 (QD1) — Basic Time-Series
=====================================================
Plots IOstat, True Utilization, and model-predicted utilization over time
for workload A42 with queue depth 1.

Usage:
    python 20_eval_workload_42_basic.py <eval_data> <models_joblib>
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
        description="Workload A42 QD1 basic evaluation."
    )
    parser.add_argument("data_path", help="Path to evaluation Excel file.")
    parser.add_argument("models_path", help="Path to saved models (.joblib).")
    parser.add_argument("--model", default="LASSO_POLY", help="Model to use.")
    parser.add_argument("--output", default="evaluation_workload_42_QD1.pdf")
    args = parser.parse_args()

    models = joblib.load(args.models_path)
    plot_time_series_evaluation(
        args.data_path, models,
        target_model=args.model,
        output_pdf=args.output,
    )
