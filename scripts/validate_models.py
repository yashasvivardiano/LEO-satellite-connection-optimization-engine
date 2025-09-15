
#!/usr/bin/env python3
"""
Validate dual-AI models using processed datasets.

Outputs:
- models/dual_ai/predictive_metrics.json (accuracy, precision/recall, confusion matrix)
- models/dual_ai/anomaly_metrics.json (reconstruction error stats, threshold, AUROC if labels available)
- models/dual_ai/reconstruction_error_distributions.png

This script assumes artifacts trained in notebooks/dual_ai_training.ipynb exist at models/dual_ai/:
- predictive_clf.keras
- autoencoder.keras
- scaler.pkl (sklearn StandardScaler for A/B features)
- manifest.json (contains feature_columns, window, horizon)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate predictive and anomaly models")
    p.add_argument("--dataset-a", default="data/processed/hybrid_v1_dataset_a.csv", help="CSV for predictive validation")
    p.add_argument("--dataset-b", default="data/processed/hybrid_v1_dataset_b.csv", help="CSV for anomaly validation (normal-only)")
    p.add_argument("--artifacts-dir", default="models/dual_ai", help="Directory with trained models and scaler")
    p.add_argument("--max-rows", type=int, default=250000, help="Optional cap on rows for fast validation")
    return p.parse_args()


def _load_model_any(path: Path):
    """Load a .keras model saved by either tf.keras (Keras 2) or keras>=3.

    We first try tf.keras for compatibility with TF 2.15; if that fails due to
    config mismatches (e.g., 'time_major' arg from Keras 3), we fallback to
    the standalone Keras 3 loader.
    """
    try:
        return tf.keras.models.load_model(path)
    except Exception:
        try:
            import keras  # type: ignore

            # Keras 3 unified loader with tolerant mode
            return keras.models.load_model(path, safe_mode=False, compile=False)
        except Exception as err:
            raise RuntimeError(f"Failed to load model at {path}: {err}")


def load_artifacts(artifacts_dir: Path) -> Tuple[Dict, object, object, object]:
    manifest = json.loads((artifacts_dir / "manifest.json").read_text())
    clf = _load_model_any(artifacts_dir / "predictive_clf.keras")
    ae = _load_model_any(artifacts_dir / "autoencoder.keras")
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    return manifest, clf, ae, scaler


def make_sequences(df: pd.DataFrame, feature_cols: List[str], label_col: str | None, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray | None]:
    data = df.copy()
    if label_col is not None and label_col not in data.columns:
        raise ValueError(f"Missing label column: {label_col}")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    for start in range(0, len(data) - window - horizon + 1):
        end = start + window
        x_window = data.iloc[start:end][feature_cols].values
        X_list.append(x_window)
        if label_col is not None:
            y_value = data.iloc[end + horizon - 1][label_col]
            # map label
            if isinstance(y_value, str):
                y_value = 0 if y_value.lower() == "wired" else 1
            y_list.append(int(y_value))
    X = np.stack(X_list) if X_list else np.empty((0, window, len(feature_cols)))
    y = np.array(y_list) if label_col is not None else None
    return X, y


def evaluate_predictive(clf: tf.keras.Model, df_a: pd.DataFrame, feature_cols: List[str], scaler, window: int, horizon: int) -> Dict:
    # Drop rows with missing labels or features
    df_a = df_a.dropna(subset=feature_cols + ["current_optimal_path"]).reset_index(drop=True)
    # Scale features
    X_features = scaler.transform(df_a[feature_cols].values)
    df_scaled = pd.DataFrame(X_features, columns=feature_cols, index=df_a.index)
    df_scaled["current_optimal_path"] = df_a["current_optimal_path"].map({"wired": 0, "satellite": 1})

    X, y = make_sequences(df_scaled, feature_cols, "current_optimal_path", window, horizon)
    if len(X) == 0:
        return {"error": "Insufficient rows to form validation sequences."}

    y_pred_prob = clf.predict(X, verbose=0)
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] == 1:
        y_pred = (y_pred_prob.ravel() >= 0.5).astype(int)
    else:
        y_pred = np.argmax(y_pred_prob, axis=1)

    accuracy = float((y_pred == y).mean())

    # Confusion matrix
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    precision = float(precision_score(y, y_pred, zero_division=0))
    recall = float(recall_score(y, y_pred, zero_division=0))
    f1 = float(f1_score(y, y_pred, zero_division=0))

    return {
        "num_sequences": int(len(X)),
        "window": window,
        "horizon": horizon,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


def reconstruction_errors(ae: tf.keras.Model, X_seq: np.ndarray) -> np.ndarray:
    X_hat = ae.predict(X_seq, verbose=0)
    err = np.mean((X_seq - X_hat) ** 2, axis=(1, 2))
    return err


def evaluate_anomaly(ae: tf.keras.Model, df_b: pd.DataFrame, feature_cols: List[str], scaler, window: int, artifacts_dir: Path) -> Dict:
    df_b = df_b.dropna(subset=feature_cols).reset_index(drop=True)
    X_features = scaler.transform(df_b[feature_cols].values)
    df_scaled = pd.DataFrame(X_features, columns=feature_cols, index=df_b.index)

    X, _ = make_sequences(df_scaled, feature_cols, None, window, 1)
    if len(X) == 0:
        return {"error": "Insufficient rows to form sequences for anomaly evaluation."}

    errs = reconstruction_errors(ae, X)
    metrics = {
        "mean_error": float(errs.mean()),
        "std_error": float(errs.std()),
        "p95_error": float(np.percentile(errs, 95)),
        "p99_error": float(np.percentile(errs, 99)),
        "num_sequences": int(len(errs)),
    }

    # If a train threshold exists, include and visualize
    try:
        train_metrics = json.loads((artifacts_dir / "anomaly_metrics.json").read_text())
        threshold = float(train_metrics.get("train_99pct_threshold", np.percentile(errs, 99)))
    except Exception:
        threshold = float(np.percentile(errs, 99))

    # Save distribution plot
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errs, bins=60, density=True, alpha=0.6, label="validation")
    ax.axvline(threshold, color="r", linestyle="--", label="threshold")
    ax.set_title("Reconstruction Error Distribution (Validation)")
    ax.set_xlabel("MSE per sequence")
    ax.grid(alpha=0.3)
    ax.legend()
    out_png = artifacts_dir / "reconstruction_error_distributions.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    metrics["threshold"] = threshold
    metrics["plot"] = str(out_png)
    return metrics


def main() -> None:
    args = parse_args()
    artifacts_dir = Path(args.artifacts_dir)

    manifest, clf, ae, scaler = load_artifacts(artifacts_dir)
    feature_cols: List[str] = list(manifest["feature_columns"])
    window: int = int(manifest.get("window", 12))
    horizon: int = int(manifest.get("horizon", 6))

    # Load datasets (sample for speed if very large)
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if args.max_rows and len(df) > args.max_rows:
            df = df.head(args.max_rows).copy()
        return df

    df_a = load_csv(args.dataset_a)
    df_b = load_csv(args.dataset_b)

    pred_metrics = evaluate_predictive(clf, df_a, feature_cols, scaler, window, horizon)
    anom_metrics = evaluate_anomaly(ae, df_b, feature_cols, scaler, window, artifacts_dir)

    # Merge and persist
    (artifacts_dir / "predictive_metrics.json").write_text(json.dumps(pred_metrics, indent=2))
    (artifacts_dir / "anomaly_metrics.json").write_text(json.dumps(anom_metrics, indent=2))

    print("Predictive metrics saved to", artifacts_dir / "predictive_metrics.json")
    print("Anomaly metrics saved to", artifacts_dir / "anomaly_metrics.json")


if __name__ == "__main__":
    main()


