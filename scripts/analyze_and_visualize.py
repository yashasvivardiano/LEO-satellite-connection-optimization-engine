#!/usr/bin/env python3
"""
Analyze and visualize hybrid datasets (A/B) with summary metrics and plots.
"""

import argparse
import json
import sys
from pathlib import Path
import time

# Force non-interactive backend to avoid GUI/prompt blocks
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze and visualize hybrid datasets")
    p.add_argument("--input", required=True, help="CSV path of dataset to analyze")
    p.add_argument("--outdir", required=True, help="Output directory for plots and report")
    p.add_argument("--title", default="Hybrid Dataset Analysis", help="Title prefix for plots")
    p.add_argument("--sample", type=int, default=0, help="Optional number of rows to sample for fast run")
    return p.parse_args()


def basic_summary(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=[np.number])
    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_columns": list(numeric.columns),
    }
    if len(numeric.columns) > 0:
        summary["describe"] = json.loads(numeric.describe(percentiles=[0.05, 0.5, 0.95]).to_json())
    return summary


def plot_time_series(df: pd.DataFrame, outdir: Path, title: str) -> list[Path]:
    paths = []
    time_col = None
    for tc in ["timestamp", "time", "datetime"]:
        if tc in df.columns:
            time_col = tc
            break

    def savefig(fig, name: str):
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        print(f"Saved {path}")
        sys.stdout.flush()

    # Latency
    if {"wired_latency_ms", "satellite_latency_ms"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = df[time_col] if time_col else np.arange(len(df))
        ax.plot(x, df["wired_latency_ms"], label="Wired", linewidth=1)
        ax.plot(x, df["satellite_latency_ms"], label="Satellite", linewidth=1)
        ax.set_title(f"{title} - Latency (ms)")
        ax.set_xlabel("Time" if time_col else "Index")
        ax.set_ylabel("ms")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, "timeseries_latency.png")

    # Jitter
    if {"wired_jitter_ms", "satellite_jitter_ms"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = df[time_col] if time_col else np.arange(len(df))
        ax.plot(x, df["wired_jitter_ms"], label="Wired", linewidth=1)
        ax.plot(x, df["satellite_jitter_ms"], label="Satellite", linewidth=1)
        ax.set_title(f"{title} - Jitter (ms)")
        ax.set_xlabel("Time" if time_col else "Index")
        ax.set_ylabel("ms")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, "timeseries_jitter.png")

    # Packet loss
    if {"wired_packet_loss_pct", "satellite_packet_loss_pct"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = df[time_col] if time_col else np.arange(len(df))
        ax.plot(x, df["wired_packet_loss_pct"], label="Wired", linewidth=1)
        ax.plot(x, df["satellite_packet_loss_pct"], label="Satellite", linewidth=1)
        ax.set_title(f"{title} - Packet Loss (%)")
        ax.set_xlabel("Time" if time_col else "Index")
        ax.set_ylabel("%")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, "timeseries_loss.png")

    # Bandwidth
    if {"wired_bandwidth_mbps", "satellite_bandwidth_mbps"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = df[time_col] if time_col else np.arange(len(df))
        ax.plot(x, df["wired_bandwidth_mbps"], label="Wired", linewidth=1)
        ax.plot(x, df["satellite_bandwidth_mbps"], label="Satellite", linewidth=1)
        ax.set_title(f"{title} - Bandwidth (Mbps)")
        ax.set_xlabel("Time" if time_col else "Index")
        ax.set_ylabel("Mbps")
        ax.legend()
        ax.grid(alpha=0.3)
        savefig(fig, "timeseries_bandwidth.png")

    return paths


def plot_distributions(df: pd.DataFrame, outdir: Path, title: str) -> list[Path]:
    paths = []
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        return paths

    def savefig(fig, name: str):
        fig.tight_layout()
        path = outdir / name
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
        print(f"Saved {path}")
        sys.stdout.flush()

    # Histograms
    fig, axes = plt.subplots(nrows=min(6, len(numeric.columns)), ncols=1, figsize=(10, 18))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, col in zip(axes, numeric.columns[: len(axes)]):
        sns.histplot(numeric[col], kde=True, ax=ax)
        ax.set_title(f"{title} - Distribution: {col}")
        ax.grid(alpha=0.3)
    savefig(fig, "distributions_top.png")

    # Boxplots comparison wired vs satellite metrics if present
    box_pairs = [
        ("wired_latency_ms", "satellite_latency_ms"),
        ("wired_jitter_ms", "satellite_jitter_ms"),
        ("wired_packet_loss_pct", "satellite_packet_loss_pct"),
        ("wired_bandwidth_mbps", "satellite_bandwidth_mbps"),
    ]
    for a, b in box_pairs:
        if a in df.columns and b in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=pd.melt(df[[a, b]], var_name="metric", value_name="value"), x="metric", y="value", ax=ax)
            ax.set_title(f"{title} - Boxplot: {a} vs {b}")
            ax.grid(alpha=0.3)
            savefig(fig, f"box_{a}_vs_{b}.png")

    return paths


def plot_correlations(df: pd.DataFrame, outdir: Path, title: str) -> list[Path]:
    paths = []
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return paths
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title(f"{title} - Correlation Matrix")
    path = outdir / "correlation_matrix.png"
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)
    print(f"Saved {path}")
    sys.stdout.flush()
    return paths


def plot_path_selection(df: pd.DataFrame, outdir: Path, title: str) -> list[Path]:
    paths = []
    if "current_optimal_path" not in df.columns:
        return paths

    # Pie / counts
    fig, ax = plt.subplots(figsize=(6, 6))
    counts = df["current_optimal_path"].value_counts(dropna=False)
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%")
    ax.set_title(f"{title} - Optimal Path Share")
    p1 = outdir / "optimal_path_share.png"
    fig.savefig(p1, dpi=200, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)
    print(f"Saved {p1}")
    sys.stdout.flush()

    # Cost comparison if available
    if {"wired_quality_cost", "satellite_quality_cost"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(12, 4))
        x = np.arange(len(df))
        ax.plot(x, df["wired_quality_cost"], label="Wired cost", linewidth=0.8)
        ax.plot(x, df["satellite_quality_cost"], label="Satellite cost", linewidth=0.8)
        ax.set_title(f"{title} - Quality Cost Over Time")
        ax.set_xlabel("Index")
        ax.set_ylabel("Cost (lower is better)")
        ax.legend()
        ax.grid(alpha=0.3)
        p2 = outdir / "quality_cost_timeseries.png"
        fig.tight_layout()
        fig.savefig(p2, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)
        print(f"Saved {p2}")
        sys.stdout.flush()

    return paths


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Loading CSV: {args.input}")
    sys.stdout.flush()
    t0 = time.time()
    df = pd.read_csv(args.input)
    if args.sample and args.sample > 0 and len(df) > args.sample:
        df = df.head(args.sample).copy()
        print(f"Sampled first {args.sample} rows for fast analysis")
    print(f"Loaded {len(df)} rows in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    report = {}
    report["summary"] = basic_summary(df)

    figs = []
    print("Plotting time series..."); sys.stdout.flush()
    figs += plot_time_series(df, outdir, args.title)
    print("Plotting distributions..."); sys.stdout.flush()
    figs += plot_distributions(df, outdir, args.title)
    print("Plotting correlations..."); sys.stdout.flush()
    figs += plot_correlations(df, outdir, args.title)
    print("Plotting path selection..."); sys.stdout.flush()
    figs += plot_path_selection(df, outdir, args.title)

    report["figures"] = [str(p) for p in figs]

    with open(outdir / "analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("Wrote report:", outdir / "analysis_report.json")
    for p in figs:
        print("Plot:", p)


if __name__ == "__main__":
    main()


