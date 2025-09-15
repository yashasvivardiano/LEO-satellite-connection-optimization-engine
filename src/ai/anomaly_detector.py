#!/usr/bin/env python3
"""
AI Anomaly Detection Module

This module integrates the trained LSTM Autoencoder model for detecting
network anomalies and critical failures in real-time.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class NetworkAnomalyDetector:
    """
    Real-time anomaly detection using trained LSTM Autoencoder.
    Detects critical network failures and unusual patterns.
    """
    
    def __init__(self, model_dir: str = "models/dual_ai"):
        """Initialize the anomaly detector with trained models."""
        self.model_dir = Path(model_dir)
        self.autoencoder = None
        self.scaler = None
        self.threshold = None
        self.feature_columns = None
        self.window_size = 12
        self.is_initialized = False
        
        # Load models and configuration
        self._load_models()
        
        # Anomaly detection state
        self.recent_windows = []
        self.anomaly_history = []
        self.detection_count = 0
        
        logger.info("Network Anomaly Detector initialized")
    
    def _load_models(self):
        """Load the trained autoencoder model and scaler."""
        try:
            # Load autoencoder model
            model_path = self.model_dir / "autoencoder.keras"
            if model_path.exists():
                self.autoencoder = tf.keras.models.load_model(str(model_path))
                logger.info(f"Loaded autoencoder from {model_path}")
            else:
                logger.error(f"Autoencoder model not found at {model_path}")
                return
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.error(f"Scaler not found at {scaler_path}")
                return
            
            # Load anomaly metrics and threshold
            metrics_path = self.model_dir / "anomaly_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                self.threshold = metrics.get('train_99pct_threshold', 0.03)
                logger.info(f"Loaded anomaly threshold: {self.threshold}")
            else:
                logger.warning("Anomaly metrics not found, using default threshold")
                self.threshold = 0.03
            
            # Load feature configuration
            manifest_path = self.model_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                self.feature_columns = manifest.get('feature_columns', [])
                self.window_size = manifest.get('window', 12)
                logger.info(f"Loaded feature config: {len(self.feature_columns)} features, window={self.window_size}")
            else:
                logger.warning("Manifest not found, using default features")
                self.feature_columns = [
                    'wired_latency_ms', 'satellite_latency_ms',
                    'wired_jitter_ms', 'satellite_jitter_ms',
                    'wired_packet_loss_pct', 'satellite_packet_loss_pct',
                    'wired_bandwidth_mbps', 'satellite_bandwidth_mbps',
                    'wired_quality_cost', 'satellite_quality_cost'
                ]
            
            self.is_initialized = True
            logger.info("Anomaly detector successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detector: {e}")
            self.is_initialized = False
    
    def detect_anomaly(self, network_data: Dict) -> Dict:
        """
        Detect anomalies in current network data.
        
        Args:
            network_data: Dictionary containing network metrics
            
        Returns:
            Dictionary with anomaly detection results
        """
        if not self.is_initialized:
            # Fallback: Use simple heuristic based on extreme values
            wired_latency = network_data.get('wired_latency_ms', 30)
            satellite_latency = network_data.get('satellite_latency_ms', 45)
            wired_loss = network_data.get('wired_packet_loss_pct', 0.015)
            satellite_loss = network_data.get('satellite_packet_loss_pct', 0.05)
            
            # Simple anomaly detection based on thresholds
            is_anomaly = (
                wired_latency > 100 or satellite_latency > 200 or
                wired_loss > 5.0 or satellite_loss > 10.0
            )
            
            # Calculate confidence based on how extreme the values are
            max_latency = max(wired_latency, satellite_latency)
            max_loss = max(wired_loss, satellite_loss)
            confidence = min(1.0, (max_latency - 50) / 200 + (max_loss - 1.0) / 10.0)
            
            return {
                "is_anomaly": is_anomaly,
                "confidence": confidence,
                "reconstruction_error": confidence * 0.1,
                "threshold": 0.1,
                "status": "Fallback heuristic"
            }
        
        try:
            # Extract features from network data
            features = self._extract_features(network_data)
            
            # Add to recent windows
            self.recent_windows.append(features)
            
            # Keep only recent windows
            if len(self.recent_windows) > self.window_size:
                self.recent_windows = self.recent_windows[-self.window_size:]
            
            # Need enough data for detection
            if len(self.recent_windows) < self.window_size:
                return {
                    "is_anomaly": False,
                    "confidence": 0.0,
                    "reconstruction_error": 0.0,
                    "threshold": self.threshold,
                    "status": "Insufficient data"
                }
            
            # Prepare data for model
            window_data = np.array(self.recent_windows[-self.window_size:])
            window_data = window_data.reshape(1, self.window_size, -1)
            
            # Get reconstruction error
            reconstruction = self.autoencoder.predict(window_data, verbose=0)
            reconstruction_error = np.mean(np.square(window_data - reconstruction))
            
            # Determine if anomaly
            is_anomaly = reconstruction_error > self.threshold
            
            # Calculate confidence (higher error = higher confidence)
            confidence = min(1.0, reconstruction_error / (self.threshold * 2))
            
            # Log anomaly detection
            if is_anomaly:
                self.detection_count += 1
                anomaly_info = {
                    "timestamp": datetime.now(),
                    "reconstruction_error": float(reconstruction_error),
                    "threshold": float(self.threshold),
                    "confidence": float(confidence),
                    "network_data": network_data
                }
                self.anomaly_history.append(anomaly_info)
                logger.warning(f"Anomaly detected! Error: {reconstruction_error:.4f}, Threshold: {self.threshold:.4f}")
            
            return {
                "is_anomaly": bool(is_anomaly),
                "confidence": float(confidence),
                "reconstruction_error": float(reconstruction_error),
                "threshold": float(self.threshold),
                "status": "Success"
            }
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return {
                "is_anomaly": False,
                "confidence": 0.0,
                "reconstruction_error": 0.0,
                "threshold": self.threshold,
                "error": str(e)
            }
    
    def _extract_features(self, network_data: Dict) -> List[float]:
        """Extract feature vector from network data."""
        features = []
        
        for col in self.feature_columns:
            if col in network_data:
                features.append(float(network_data[col]))
            else:
                logger.warning(f"Feature {col} not found in network data")
                features.append(0.0)
        
        return features
    
    def get_anomaly_summary(self) -> Dict:
        """Get summary of anomaly detection history."""
        return {
            "total_detections": self.detection_count,
            "recent_anomalies": len([a for a in self.anomaly_history 
                                   if (datetime.now() - a["timestamp"]).total_seconds() < 3600]),
            "threshold": self.threshold,
            "is_initialized": self.is_initialized,
            "window_size": self.window_size,
            "feature_count": len(self.feature_columns)
        }
    
    def get_recent_anomalies(self, hours: int = 1) -> List[Dict]:
        """Get recent anomalies within specified time window."""
        cutoff_time = datetime.now() - pd.Timedelta(hours=hours)
        return [a for a in self.anomaly_history if a["timestamp"] > cutoff_time]
    
    def reset_history(self):
        """Reset anomaly detection history."""
        self.anomaly_history = []
        self.detection_count = 0
        logger.info("Anomaly detection history reset")
    
    def update_threshold(self, new_threshold: float):
        """Update the anomaly detection threshold."""
        old_threshold = self.threshold
        self.threshold = new_threshold
        logger.info(f"Anomaly threshold updated: {old_threshold:.4f} -> {new_threshold:.4f}")

class CriticalFailureDetector:
    """
    Specialized detector for critical network failures.
    Uses multiple criteria to identify severe network issues.
    """
    
    def __init__(self):
        """Initialize critical failure detector."""
        self.failure_criteria = {
            "latency_threshold": 200.0,  # ms
            "jitter_threshold": 50.0,    # ms
            "loss_threshold": 10.0,      # %
            "bandwidth_threshold": 10.0  # mbps
        }
        
        self.failure_history = []
        logger.info("Critical Failure Detector initialized")
    
    def detect_critical_failure(self, network_data: Dict, path: str) -> Dict:
        """
        Detect critical failures in network path.
        
        Args:
            network_data: Network metrics data
            path: Network path ('wired' or 'satellite')
            
        Returns:
            Dictionary with critical failure detection results
        """
        try:
            # Extract path-specific metrics
            prefix = f"{path}_"
            latency = network_data.get(f"{prefix}latency_ms", 0)
            jitter = network_data.get(f"{prefix}jitter_ms", 0)
            loss = network_data.get(f"{prefix}packet_loss_pct", 0)
            bandwidth = network_data.get(f"{prefix}bandwidth_mbps", 0)
            
            # Check failure criteria
            failures = []
            
            if latency > self.failure_criteria["latency_threshold"]:
                failures.append(f"High latency: {latency:.1f}ms")
            
            if jitter > self.failure_criteria["jitter_threshold"]:
                failures.append(f"High jitter: {jitter:.1f}ms")
            
            if loss > self.failure_criteria["loss_threshold"]:
                failures.append(f"High packet loss: {loss:.1f}%")
            
            if bandwidth < self.failure_criteria["bandwidth_threshold"]:
                failures.append(f"Low bandwidth: {bandwidth:.1f}mbps")
            
            is_critical = len(failures) > 0
            
            if is_critical:
                failure_info = {
                    "timestamp": datetime.now(),
                    "path": path,
                    "failures": failures,
                    "metrics": {
                        "latency": latency,
                        "jitter": jitter,
                        "loss": loss,
                        "bandwidth": bandwidth
                    }
                }
                self.failure_history.append(failure_info)
                logger.critical(f"Critical failure detected on {path}: {failures}")
            
            return {
                "is_critical": is_critical,
                "failures": failures,
                "path": path,
                "severity": len(failures),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in critical failure detection: {e}")
            return {
                "is_critical": False,
                "failures": [],
                "path": path,
                "error": str(e)
            }
    
    def get_failure_summary(self) -> Dict:
        """Get summary of critical failures."""
        recent_failures = [f for f in self.failure_history 
                          if (datetime.now() - f["timestamp"]).total_seconds() < 3600]
        
        return {
            "total_failures": len(self.failure_history),
            "recent_failures": len(recent_failures),
            "criteria": self.failure_criteria
        }

def create_anomaly_detector(model_dir: str = "models/dual_ai") -> NetworkAnomalyDetector:
    """Create a new anomaly detector instance."""
    return NetworkAnomalyDetector(model_dir)

def create_critical_failure_detector() -> CriticalFailureDetector:
    """Create a new critical failure detector instance."""
    return CriticalFailureDetector()