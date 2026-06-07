"""Training and evaluation pipeline for the manual-feature SVM family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..features import CENTROID_FEATURE, FINAL_FEATURES_20, add_centroid_distance, dataframe_to_matrix
from ..metrics import evaluate_binary_scores


@dataclass(frozen=True)
class SVMConfig:
    """One operational SVM configuration from the final comparison."""

    name: str
    feature_columns: tuple[str, ...]
    threshold: float
    calibrated: bool = False
    c_value: float = 10.0
    gamma: str | float = "scale"


SVM_BASE_FINAL = SVMConfig("SVM-base-final", tuple(FINAL_FEATURES_20), threshold=0.0)
SVM_CONSERVATIVE = SVMConfig("SVM-conservative-centroid", tuple(FINAL_FEATURES_20 + [CENTROID_FEATURE]), threshold=0.0)
SVM_AUGMENTED = SVMConfig("SVM-augmented", tuple(FINAL_FEATURES_20 + [CENTROID_FEATURE]), threshold=0.0)
SVM_CALIBRATED = SVMConfig("SVM-calibrated", tuple(FINAL_FEATURES_20 + [CENTROID_FEATURE]), threshold=0.5, calibrated=True)
SVM_SENSITIVE = SVMConfig("SVM-sensitive", tuple(FINAL_FEATURES_20 + [CENTROID_FEATURE]), threshold=0.15, calibrated=True)


def fit_scaler(train_frame: pd.DataFrame, feature_columns: list[str] = FINAL_FEATURES_20) -> StandardScaler:
    """Fit a StandardScaler on training features only."""
    scaler = StandardScaler()
    scaler.fit(dataframe_to_matrix(train_frame, feature_columns))
    return scaler


def compute_positive_centroid(X_train_scaled: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute centroid and per-feature spread from positive training windows."""
    positives = X_train_scaled[np.asarray(y_train, dtype=int) == 1]
    if len(positives) == 0:
        raise ValueError("At least one positive training window is required.")
    return positives.mean(axis=0), positives.std(axis=0)


def build_feature_matrix(
    frame: pd.DataFrame,
    scaler: StandardScaler,
    feature_columns: list[str] = FINAL_FEATURES_20,
    centroid: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Scale manual features and optionally append centroid distance."""
    X_scaled = scaler.transform(dataframe_to_matrix(frame, feature_columns))
    if centroid is None:
        return X_scaled
    center, spread = centroid
    return add_centroid_distance(X_scaled, center, spread)


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: SVMConfig,
    sample_weight: np.ndarray | None = None,
) -> SVC | CalibratedClassifierCV:
    """Train one SVM configuration."""
    base = SVC(
        kernel="rbf",
        C=config.c_value,
        gamma=config.gamma,
        class_weight="balanced",
        random_state=42,
    )
    if not config.calibrated:
        base.fit(X_train, y_train, sample_weight=sample_weight)
        return base

    calibrated = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    calibrated.fit(X_train, y_train, sample_weight=sample_weight)
    return calibrated


def score_svm(model: SVC | CalibratedClassifierCV, X: np.ndarray) -> np.ndarray:
    """Return comparable positive-class scores for calibrated or uncalibrated SVMs."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate_model(
    model: SVC | CalibratedClassifierCV,
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate a trained SVM on one split."""
    scores = score_svm(model, X)
    return evaluate_binary_scores(split_name, y, scores, threshold).to_dict()


def save_svm_bundle(
    output_path: str | Path,
    model: SVC | CalibratedClassifierCV,
    scaler: StandardScaler,
    config: SVMConfig,
    feature_columns: list[str],
    centroid: tuple[np.ndarray, np.ndarray] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Serialize model, preprocessing objects, and the inference contract."""
    bundle = {
        "model": model,
        "scaler": scaler,
        "config": config,
        "feature_columns": list(feature_columns),
        "centroid": centroid,
        "metadata": metadata or {},
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
