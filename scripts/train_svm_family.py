#!/usr/bin/env python
"""Train one final SVM variant from feature CSV files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from birdsong_detector.classical.svm_pipeline import (
    SVM_AUGMENTED,
    SVM_BASE_FINAL,
    SVM_CALIBRATED,
    SVM_CONSERVATIVE,
    SVM_SENSITIVE,
    build_feature_matrix,
    compute_positive_centroid,
    evaluate_model,
    fit_scaler,
    save_svm_bundle,
    train_svm,
)
from birdsong_detector.features import FINAL_FEATURES_20


CONFIGS = {
    "base": SVM_BASE_FINAL,
    "conservative": SVM_CONSERVATIVE,
    "augmented": SVM_AUGMENTED,
    "calibrated": SVM_CALIBRATED,
    "sensitive": SVM_SENSITIVE,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--val-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--field-csv", type=Path)
    parser.add_argument("--variant", choices=sorted(CONFIGS), default="augmented")
    parser.add_argument("--output", default=Path("outputs/models/svm_model.joblib"), type=Path)
    return parser.parse_args()


def _load(path: Path | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def main() -> None:
    args = parse_args()
    config = CONFIGS[args.variant]
    train_df = pd.read_csv(args.train_csv)
    y_train = train_df["label"].to_numpy(dtype=int)

    scaler = fit_scaler(train_df, FINAL_FEATURES_20)
    X_train_20 = build_feature_matrix(train_df, scaler, FINAL_FEATURES_20)
    centroid = None
    X_train = X_train_20
    if len(config.feature_columns) == 21:
        centroid = compute_positive_centroid(X_train_20, y_train)
        X_train = build_feature_matrix(train_df, scaler, FINAL_FEATURES_20, centroid=centroid)

    sample_weight = train_df["sample_weight"].to_numpy(dtype=float) if "sample_weight" in train_df.columns else None
    model = train_svm(X_train, y_train, config, sample_weight=sample_weight)
    save_svm_bundle(args.output, model, scaler, config, FINAL_FEATURES_20, centroid=centroid)

    rows = []
    for split_name, split_path in [("validation", args.val_csv), ("internal_test", args.test_csv), ("field_holdout", args.field_csv)]:
        frame = _load(split_path)
        if frame is None:
            continue
        X = build_feature_matrix(frame, scaler, FINAL_FEATURES_20, centroid=centroid)
        rows.append(evaluate_model(model, split_name, X, frame["label"].to_numpy(dtype=int), config.threshold))

    if rows:
        metrics_path = args.output.with_suffix(".metrics.csv")
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"Metrics written to {metrics_path.resolve()}")
    print(f"Model written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
