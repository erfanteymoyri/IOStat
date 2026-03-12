"""
Evaluation — Workload A669 (QD1) — Model Comparison
=====================================================
Usage:
    python 29_eval_workload_669_comparison.py <eval_data> <old_models> <new_models>
"""
import argparse, os, sys, joblib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from evaluation_utils import plot_comparison_evaluation

HARDCODED_PREDICTIONS_A669 = [
    2.128710971, 6.65895429, 29.50903599, 10.3838818, 10.45484904, 51.98200351,
    2.149472118, 6.154247694, 11.42526472, 29.39865835, 17.01920075, 15.69285714,
    5, 5.075, 5.335, 23.51384468, 5.071428571, 4.963571429, 4.766428571, 4.715,
    4.351428571, 4.35, 4.26, 4.285, 4.326428571, 4.31, 4.422142857, 4.322142857,
    4.338571429, 4.385, 4.372142857, 4.422857143, 4.256428571, 4.277142857, 4.340714286,
    4.367857143, 4.434285714, 4.391428571, 4.5, 4.532142857, 4.457142857, 4.534285714,
    4.502142857, 4.421428571, 4.387857143, 4.324285714, 4.359285714, 4.432857143,
    5.145714286, 5.446428571, 5.364285714,
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workload A669 QD1 model comparison.")
    parser.add_argument("data_path"); parser.add_argument("old_models"); parser.add_argument("new_models")
    parser.add_argument("--output", default="evaluation_workload_669_QD1_comparison.pdf")
    args = parser.parse_args()
    old_models = joblib.load(args.old_models)
    plot_comparison_evaluation(
        args.data_path, old_models=old_models, new_models_path=args.new_models,
        old_model_name="LASSO_POLY", new_model_name="RANDOM_FOREST",
        hardcoded_predictions=HARDCODED_PREDICTIONS_A669, output_pdf=args.output,
    )
