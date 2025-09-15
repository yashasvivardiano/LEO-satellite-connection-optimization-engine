#!/usr/bin/env python3
"""
CLI to process hybrid telemetry into Dataset A (predictive) and Dataset B (anomaly).
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure `src` is importable without triggering package-level imports in src.__init__
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

try:
    # Prefer normal package import if utils package is healthy
    from utils.data_processor import DataProcessor, SmoothingConfig, LabelingWeights
except Exception:
    # Fallback: import the module directly from file to avoid utils.__init__ side effects
    import importlib.util

    dp_path = SRC_DIR / "utils" / "data_processor.py"
    spec = importlib.util.spec_from_file_location("_dp_module", str(dp_path))
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    import sys as _sys
    _sys.modules[spec.name] = module  # type: ignore[union-attr]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    DataProcessor = module.DataProcessor
    SmoothingConfig = module.SmoothingConfig
    LabelingWeights = module.LabelingWeights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build datasets for predictive and anomaly models")
    parser.add_argument("--input", required=True, help="Path to input hybrid CSV")
    parser.add_argument("--output-dir", required=True, help="Directory to write outputs")
    parser.add_argument("--prefix", default="hybrid_processed", help="Filename prefix for outputs")
    parser.add_argument("--smooth-window", type=int, default=21, help="Savitzky-Golay window length (odd)")
    parser.add_argument("--smooth-poly", type=int, default=2, help="Savitzky-Golay polynomial order")
    parser.add_argument("--w-latency", type=float, default=0.4, help="Weight for latency in quality cost")
    parser.add_argument("--w-jitter", type=float, default=0.25, help="Weight for jitter in quality cost")
    parser.add_argument("--w-loss", type=float, default=0.25, help="Weight for packet loss in quality cost")
    parser.add_argument("--w-bandwidth", type=float, default=0.10, help="Weight for bandwidth in quality cost")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    smoothing = SmoothingConfig(window_length=args.smooth_window, polyorder=args.smooth_poly)
    weights = LabelingWeights(
        latency=args.w_latency,
        jitter=args.w_jitter,
        packet_loss=args.w_loss,
        bandwidth=args.w_bandwidth,
    )

    processor = DataProcessor(smoothing=smoothing, weights=weights)

    outputs = processor.process(
        csv_path=Path(args.input),
        output_dir=Path(args.output_dir),
        filename_prefix=args.prefix,
    )

    print("Dataset A:", outputs["dataset_a"])
    print("Dataset B:", outputs["dataset_b"])


if __name__ == "__main__":
    main()


