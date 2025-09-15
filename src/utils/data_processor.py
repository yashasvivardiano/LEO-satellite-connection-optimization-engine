"""
Data processing utilities for hybrid wired/satellite datasets.

This module provides the `DataProcessor` class, which loads raw hybrid
telemetry, cleans and smooths noisy signals, engineers model-ready
features (including the `current_optimal_path` label), and splits outputs
into two datasets:

- Dataset A: Supervised predictive dataset (full labeled data)
- Dataset B: Unsupervised anomaly dataset (filtered to normal-only data)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import logging


logger = logging.getLogger(__name__)


@dataclass
class SmoothingConfig:
    """Configuration for Savitzky-Golay smoothing of numeric columns."""
    window_length: int = 21
    polyorder: int = 2
    columns: Tuple[str, ...] = (
        "wired_jitter_ms",
        "satellite_jitter_ms",
        "wired_latency_ms",
        "satellite_latency_ms",
        "wired_packet_loss_pct",
        "satellite_packet_loss_pct",
    )


@dataclass
class LabelingWeights:
    """Weights for composite path quality scoring."""

    latency: float = 0.4
    jitter: float = 0.25
    packet_loss: float = 0.25
    bandwidth: float = 0.10


class DataProcessor:
    """End-to-end processing for hybrid telemetry datasets."""

    def __init__(
        self,
        smoothing: Optional[SmoothingConfig] = None,
        weights: Optional[LabelingWeights] = None,
        timestamp_column: str = "timestamp",
    ) -> None:
        self.smoothing = smoothing or SmoothingConfig()
        self.weights = weights or LabelingWeights()
        self.timestamp_column = timestamp_column

    # ------------------------------ Public API ------------------------------
    def load(self, csv_path: str | Path) -> pd.DataFrame:
        path = Path(csv_path)
        logger.info("Loading dataset: %s", path)
        df = pd.read_csv(
            path,
            parse_dates=[self.timestamp_column],
        )
        if self.timestamp_column in df.columns:
            df = df.sort_values(self.timestamp_column).reset_index(drop=True)
        return df

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Basic NA handling for known numeric columns
        numeric_cols = [
            c
            for c in df.columns
            if df[c].dtype.kind in ("i", "u", "f") and c != self.timestamp_column
        ]

        # Replace infs and drop rows that are completely invalid
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

        # Simple forward/backward fill for intermittent gaps, then drop any remaining
        df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
        df = df.dropna(subset=numeric_cols)

        # Clip impossible negatives for latency/jitter/loss
        for col in [
            "wired_latency_ms",
            "satellite_latency_ms",
            "wired_jitter_ms",
            "satellite_jitter_ms",
            "wired_packet_loss_pct",
            "satellite_packet_loss_pct",
        ]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)

        return df

    def smooth(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        win = self.smoothing.window_length
        poly = self.smoothing.polyorder
        cols = [c for c in self.smoothing.columns if c in df.columns]

        if len(df) < max(win, poly + 2):
            # Not enough samples to smooth; return as-is
            return df

        # Ensure odd window and <= length
        if win % 2 == 0:
            win += 1
        if win > len(df):
            win = len(df) if len(df) % 2 == 1 else len(df) - 1
        if win < poly + 2:
            win = poly + 3

        for col in cols:
            try:
                df[f"{col}_sm"] = savgol_filter(df[col].values, window_length=win, polyorder=poly)
            except Exception:
                # Fallback to rolling mean if savgol fails
                df[f"{col}_sm"] = (
                    df[col].rolling(window=max(5, min(51, len(df)//20)), center=True).mean().bfill().ffill()
                )
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Use smoothed columns when available
        def pick(col: str) -> str:
            return f"{col}_sm" if f"{col}_sm" in df.columns else col

        # Robust normalization using 5th-95th percentiles
        def robust_norm(series: pd.Series) -> pd.Series:
            lo, hi = np.nanpercentile(series, [5, 95])
            if hi - lo <= 0:
                return pd.Series(np.zeros(len(series)), index=series.index)
            return (series - lo) / (hi - lo)

        # Build cost for each path (lower is better)
        lat_w = robust_norm(df[pick("wired_latency_ms")]) if pick("wired_latency_ms") in df else 0
        lat_s = robust_norm(df[pick("satellite_latency_ms")]) if pick("satellite_latency_ms") in df else 0
        jit_w = robust_norm(df[pick("wired_jitter_ms")]) if pick("wired_jitter_ms") in df else 0
        jit_s = robust_norm(df[pick("satellite_jitter_ms")]) if pick("satellite_jitter_ms") in df else 0
        los_w = robust_norm(df[pick("wired_packet_loss_pct")]) if pick("wired_packet_loss_pct") in df else 0
        los_s = robust_norm(df[pick("satellite_packet_loss_pct")]) if pick("satellite_packet_loss_pct") in df else 0
        bw_w = robust_norm(-df[pick("wired_bandwidth_mbps")]) if pick("wired_bandwidth_mbps") in df else 0
        bw_s = robust_norm(-df[pick("satellite_bandwidth_mbps")]) if pick("satellite_bandwidth_mbps") in df else 0

        w = self.weights
        cost_wired = w.latency * lat_w + w.jitter * jit_w + w.packet_loss * los_w + w.bandwidth * bw_w
        cost_sat = w.latency * lat_s + w.jitter * jit_s + w.packet_loss * los_s + w.bandwidth * bw_s

        df["wired_quality_cost"] = cost_wired
        df["satellite_quality_cost"] = cost_sat
        df["current_optimal_path"] = np.where(cost_wired <= cost_sat, "wired", "satellite")

        # Optional: mark predictable windows if `event_type` exists, else derive simple flags
        if "event_type" in df.columns:
            df["is_predictable_window"] = df["event_type"].isin([
                "wired_congestion",
                "satellite_handoff",
            ])
        else:
            df["is_predictable_window"] = False

        return df

    def split_datasets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return Dataset A (supervised) and Dataset B (normal-only)."""

        dataset_a = df.copy()

        # Define normal for anomaly training
        if "event_type" in df.columns:
            normal_mask = df["event_type"].fillna("normal").str.lower().eq("normal")
        else:
            normal_mask = pd.Series(True, index=df.index)

        # Further filter by reasonable operating envelope to reduce subtle anomalies
        def within(series: pd.Series, p_lo: float, p_hi: float) -> pd.Series:
            lo, hi = np.nanpercentile(series, [p_lo, p_hi])
            return (series >= lo) & (series <= hi)

        env_mask = pd.Series(True, index=df.index)
        for col in [
            "wired_latency_ms",
            "satellite_latency_ms",
            "wired_jitter_ms",
            "satellite_jitter_ms",
            "wired_packet_loss_pct",
            "satellite_packet_loss_pct",
        ]:
            if col in df.columns:
                env_mask &= within(df[col], 2, 98)

        dataset_b = df[normal_mask & env_mask].copy()
        return dataset_a, dataset_b

    def process(
        self,
        csv_path: str | Path,
        output_dir: str | Path,
        filename_prefix: str = "hybrid_processed",
    ) -> Dict[str, str]:
        df = self.load(csv_path)
        df = self.clean(df)
        df = self.smooth(df)
        df = self.engineer_features(df)
        dataset_a, dataset_b = self.split_datasets(df)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        a_path = out_dir / f"{filename_prefix}_dataset_a.csv"
        b_path = out_dir / f"{filename_prefix}_dataset_b.csv"
        dataset_a.to_csv(a_path, index=False)
        dataset_b.to_csv(b_path, index=False)

        return {"dataset_a": str(a_path), "dataset_b": str(b_path)}

