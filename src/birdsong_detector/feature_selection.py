"""Correlation-based feature selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CorrelationPair:
    """One highly correlated pair of candidate features."""

    feature_a: str
    feature_b: str
    correlation: float


def correlation_matrix(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Compute a Pearson correlation matrix for numeric candidate features."""
    data = frame.loc[:, feature_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    if data.empty:
        raise ValueError("No valid rows remain after removing NaN/inf values.")
    return data.corr(method="pearson")


def highly_correlated_pairs(corr: pd.DataFrame, threshold: float = 0.95) -> list[CorrelationPair]:
    """List feature pairs whose absolute Pearson correlation exceeds threshold."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    pairs: list[CorrelationPair] = []
    columns = list(corr.columns)
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1 :]:
            value = float(corr.loc[feature_a, feature_b])
            if abs(value) >= threshold:
                pairs.append(CorrelationPair(feature_a, feature_b, value))
    return sorted(pairs, key=lambda item: abs(item.correlation), reverse=True)


def suggest_redundant_features(
    pairs: list[CorrelationPair],
    protected_features: set[str] | None = None,
) -> list[str]:
    """Suggest a deterministic drop list while preserving protected features."""
    protected = protected_features or set()
    dropped: set[str] = set()
    for pair in pairs:
        a_available = pair.feature_a not in protected and pair.feature_a not in dropped
        b_available = pair.feature_b not in protected and pair.feature_b not in dropped
        if a_available and b_available:
            dropped.add(pair.feature_b)
        elif b_available and pair.feature_a in protected:
            dropped.add(pair.feature_b)
        elif a_available and pair.feature_b in protected:
            dropped.add(pair.feature_a)
    return sorted(dropped)


def save_correlation_report(
    frame: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
    threshold: float = 0.95,
    protected_features: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write matrix and redundant-pair tables for the feature selection tool."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    corr = correlation_matrix(frame, feature_columns)
    pairs = highly_correlated_pairs(corr, threshold=threshold)
    drop_list = suggest_redundant_features(pairs, protected_features=protected_features)

    pairs_df = pd.DataFrame(
        [
            {
                "feature_a": item.feature_a,
                "feature_b": item.feature_b,
                "correlation": item.correlation,
                "abs_correlation": abs(item.correlation),
                "suggested_drop": item.feature_b if item.feature_b in drop_list else item.feature_a if item.feature_a in drop_list else "",
            }
            for item in pairs
        ]
    )
    corr.to_csv(output / "feature_correlation_matrix.csv")
    pairs_df.to_csv(output / "high_correlation_pairs.csv", index=False)
    pd.Series(drop_list, name="feature").to_csv(output / "suggested_redundant_features.csv", index=False)
    return corr, pairs_df


def plot_correlation_heatmap(corr: pd.DataFrame, output_path: str | Path) -> None:
    """Save a compact heatmap of the feature-correlation matrix."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title("Final acoustic-feature correlation")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
