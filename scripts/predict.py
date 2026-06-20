#!/usr/bin/env python3
"""
BirdCLEF 2026 — Inference Pipeline

Usage:
    python scripts/predict.py --test-dir /path/to/test_soundscapes \\
                              --onnx-dir ./runs/latest \\
                              --output submission.csv

Loads ONNX fold models, runs inference on test soundscapes,
and writes a submission CSV.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import Config
from src.inference.predict import InferenceEngine
from src.inference.submission import write_submission


def parse_args():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 Inference Pipeline")
    parser.add_argument("--test-dir", type=str, default=None,
                        help="Path to test_soundscapes directory")
    parser.add_argument("--onnx-dir", type=str, default="./runs/latest",
                        help="Directory containing ONNX fold models")
    parser.add_argument("--output", type=str, default="submission.csv",
                        help="Output submission CSV path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test files (for debugging)")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    onnx_dir = Path(args.onnx_dir)
    if not onnx_dir.exists():
        print(f"ONNX directory not found: {onnx_dir}")
        sys.exit(1)

    if args.test_dir:
        cfg.comp_dir = str(Path(args.test_dir).parent.parent)
        test_dir = Path(args.test_dir)
    else:
        test_dir = cfg.test_dir

    print("=" * 60)
    print("BirdCLEF 2026 — Inference")
    print("=" * 60)
    print(f"ONNX models: {onnx_dir}")
    print(f"Test files:  {test_dir}")

    engine = InferenceEngine(cfg, onnx_dir=onnx_dir)
    print(f"Loaded {engine.num_folds} ONNX fold(s)")

    t0 = time.time()
    row_ids, predictions = engine.predict_all(test_dir, limit=args.limit)
    elapsed = time.time() - t0
    print(f"\nInference: {len(row_ids)} rows, {elapsed:.1f}s total")

    write_submission(row_ids, predictions, engine.primary_labels, args.output)
    print(f"Submission written to: {args.output}")


if __name__ == "__main__":
    main()
