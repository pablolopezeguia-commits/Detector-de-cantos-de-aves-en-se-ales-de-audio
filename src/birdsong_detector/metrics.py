"""Evaluation helpers shared by SVM and YAMNet models."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryMetrics:
    """Binary song/no-song metrics for one split and one operating threshold."""

    split: str
    windows: int
    positives: int
    negatives: int
    threshold: float
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    balanced_accuracy: float | None
    precision: float
    recall: float
    f1: float
    specificity: float
    fpr: float
    roc_auc: float | None
    average_precision: float | None

    def to_dict(self) -> dict[str, float | int | str | None]:
        return asdict(self)


def evaluate_binary_scores(split: str, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> BinaryMetrics:
    """Evaluate continuous scores after applying one operating threshold."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    y_pred = (scores >= float(threshold)).astype(int)
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    specificity = float(tn / max(tn + fp, 1))
    fpr = float(fp / max(fp + tn, 1))
    multi_class = len(np.unique(y_true)) > 1
    return BinaryMetrics(
        split=split,
        windows=int(len(y_true)),
        positives=positives,
        negatives=negatives,
        threshold=float(threshold),
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=float((tp + tn) / max(len(y_true), 1)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)) if multi_class else None,
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity=specificity,
        fpr=fpr,
        roc_auc=float(roc_auc_score(y_true, scores)) if multi_class else None,
        average_precision=float(average_precision_score(y_true, scores)) if multi_class else None,
    )


def threshold_grid(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    """Evaluate a grid of thresholds and return a sortable table."""
    if thresholds is None:
        thresholds = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 201)))
    rows = [evaluate_binary_scores("validation", y_true, scores, float(thr)).to_dict() for thr in thresholds]
    return pd.DataFrame(rows).sort_values(["f1", "recall", "precision"], ascending=[False, False, False])
