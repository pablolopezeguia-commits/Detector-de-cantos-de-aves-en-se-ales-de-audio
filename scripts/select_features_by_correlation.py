#!/usr/bin/env python
"""Run the correlation-based acoustic feature selection report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from birdsong_detector.feature_selection import plot_correlation_heatmap, save_correlation_report
from birdsong_detector.features import FINAL_FEATURES_20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-csv", required=True, type=Path, help="CSV table with candidate acoustic features.")
    parser.add_argument("--output-dir", default=Path("outputs/feature_selection"), type=Path)
    parser.add_argument("--threshold", default=0.95, type=float, help="Absolute Pearson correlation threshold.")
    parser.add_argument("--feature", action="append", dest="features", help="Feature column to include. Defaults to the final 20.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.features_csv)
    features = args.features or FINAL_FEATURES_20
    corr, _pairs = save_correlation_report(frame, features, args.output_dir, threshold=args.threshold)
    plot_correlation_heatmap(corr, args.output_dir / "feature_correlation_heatmap.png")
    print(f"Report written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
