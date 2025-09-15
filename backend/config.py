"""
Backend configuration for AI Network Stabilization Dashboard
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
BACKEND_DIR = BASE_DIR / "backend"

# Server configuration
SERVER_CONFIG = {
    "host": "localhost",
    "port": 5000,
    "debug": True,
    "threaded": True
}

# API configuration
API_CONFIG = {
    "cors_origins": ["http://localhost:5000", "http://127.0.0.1:5000"],
    "max_content_length": 16 * 1024 * 1024,  # 16MB
    "timeout": 30
}

# Simulation configuration
SIMULATION_CONFIG = {
    "update_interval": 3,  # seconds
    "max_data_points": 15,
    "max_alerts": 10,
    "auto_start": True
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/dashboard.log"
}

# AI Model paths
MODEL_PATHS = {
    "anomaly_detector": "models/dual_ai/autoencoder.keras",
    "network_analyzer": "models/dual_ai/predictive_clf.keras",
    "scaler": "models/dual_ai/scaler.pkl",
    "metrics": "models/dual_ai/anomaly_metrics.json"
}

# Network simulation parameters
NETWORK_CONFIG = {
    "wired_latency_range": (25, 40),
    "satellite_latency_range": (35, 60),
    "wired_loss_range": (0.005, 0.02),
    "satellite_loss_range": (0.02, 0.08),
    "wired_bandwidth_range": (900, 1000),
    "satellite_bandwidth_range": (700, 900),
    "quality_cost_base": 1.0
}
