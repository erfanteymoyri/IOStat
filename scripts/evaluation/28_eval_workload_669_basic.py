"""
Evaluation — Workload A669 (QD1) — Basic Time-Series
======================================================
Usage:
    python 28_eval_workload_669_basic.py <eval_data> <models_joblib>
"""
import argparse, os, sys, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_time_series_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workload A669 QD1 basic evaluation.")
    parser.add_argument("data_path"); parser.add_argument("models_path")
    parser.add_argument("--model", default="LASSO_POLY")
    parser.add_argument("--output", default="evaluation_workload_669_QD1.pdf")
    args = parser.parse_args()
    models = joblib.load(args.models_path)
    plot_time_series_evaluation(args.data_path, models, target_model=args.model, output_pdf=args.output)
