#!/usr/bin/env python
"""Extract final manual acoustic features from a folder of audio files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from birdsong_detector.audio import iter_complete_windows, read_mono
from birdsong_detector.features import extract_features_from_window


SUPPORTED = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None)
    parser.add_argument("--window-s", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pattern = "**/*" if args.recursive else "*"
    rows = []
    for path in sorted(p for p in args.audio_dir.glob(pattern) if p.suffix.lower() in SUPPORTED):
        audio, sr = read_mono(path)
        for window_index, start_s, end_s, chunk in iter_complete_windows(audio, sr, args.window_s, args.overlap):
            row = {
                "filepath": str(path),
                "recording_id": path.stem,
                "window_index": window_index,
                "t_start": round(start_s, 3),
                "t_end": round(end_s, 3),
                "sample_rate": sr,
            }
            if args.label is not None:
                row["label"] = int(args.label)
            row.update(extract_features_from_window(chunk, sr))
            rows.append(row)

    output = args.output_csv
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} windows to {output.resolve()}")


if __name__ == "__main__":
    main()
