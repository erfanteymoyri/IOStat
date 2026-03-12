"""
Execute Model Training
=======================
Trains the ML models on the full training dataset (AllDevices.xlsx) and
stores the resulting model dictionary in memory for use by subsequent
evaluation scripts.  Can also save models to disk with joblib.

Usage:
    python 13_execute_training.py <train_data_path> [--save <output_path>]
"""

import argparse
import os
import sys
import warnings

import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import FEATURES
from utils import read_and_prep_data, engineer_features, get_base_pipelines

warnings.filterwarnings("ignore")


def train_all_models(train_path, features):
    """Train one set of models per device and return the model dictionary."""
    train_df = read_and_prep_data(train_path, is_test_file=False)
    df_train_eng = engineer_features(train_df)

    print("\n--- Starting Model Training ---")
    models = {}
    for dev in df_train_eng["device"].unique():
        if dev == "unknown":
            continue
        sub = df_train_eng[df_train_eng["device"] == dev]
        if len(sub) < 5:
            continue

        X = sub[features]
        y = sub["total_iops"]

        dev_models = get_base_pipelines()
        for m in dev_models.values():
            m.fit(X, y)

        models[dev] = dev_models
        print(f"  Trained models for: {dev}")

    print(f"Trained devices: {list(models.keys())}")
    return models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML models on device data.")
    parser.add_argument("train_data", help="Path to training data (e.g. AllDevices.xlsx).")
    parser.add_argument(
        "--save", default=None,
        help="Save trained models to this path (e.g. saved_models.joblib).",
    )
    args = parser.parse_args()

    models = train_all_models(args.train_data, FEATURES)

    if args.save:
        joblib.dump(models, args.save)
        print(f"Models saved to: {args.save}")
