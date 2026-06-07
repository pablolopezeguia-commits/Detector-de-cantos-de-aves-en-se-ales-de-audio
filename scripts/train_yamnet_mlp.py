#!/usr/bin/env python
"""Train the YAMNet + MLP detector from an audio metadata CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--yamnet-source", default="https://tfhub.dev/google/yamnet/1")
    parser.add_argument("--output-model", default=Path("outputs/models/yamnet_final_mlp.keras"), type=Path)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--path-column", default="filepath")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from birdsong_detector.deep_learning.yamnet_mlp import extract_embedding_table, load_yamnet, train_mlp

    metadata = pd.read_csv(args.metadata_csv)
    if "split" not in metadata.columns:
        raise ValueError("metadata CSV must contain a split column with train and val values.")

    if args.embedding_cache and args.embedding_cache.exists():
        data = np.load(args.embedding_cache, allow_pickle=True)
        X_train, y_train, w_train = data["X_train"], data["y_train"], data["w_train"]
        X_val, y_val = data["X_val"], data["y_val"]
    else:
        yamnet = load_yamnet(args.yamnet_source)
        train_df = metadata[metadata["split"] == "train"].copy()
        val_df = metadata[metadata["split"] == "val"].copy()
        X_train, y_train, w_train = extract_embedding_table(train_df, yamnet, path_column=args.path_column)
        X_val, y_val, _w_val = extract_embedding_table(val_df, yamnet, path_column=args.path_column)
        if args.embedding_cache:
            args.embedding_cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.embedding_cache, X_train=X_train, y_train=y_train, w_train=w_train, X_val=X_val, y_val=y_val)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    train_mlp(X_train, y_train, X_val, y_val, sample_weight=w_train, output_path=args.output_model)
    print(f"Model written to {args.output_model.resolve()}")


if __name__ == "__main__":
    main()
