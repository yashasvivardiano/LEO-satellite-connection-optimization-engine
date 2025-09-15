"""
AI and machine learning components for satellite network analysis.

This module contains AI-powered components for predictive maintenance,
anomaly detection, and network optimization.
"""

from .network_analyzer import PredictiveNetworkAnalyzer, create_network_analyzer
from .anomaly_detector import NetworkAnomalyDetector, CriticalFailureDetector, create_anomaly_detector, create_critical_failure_detector

# Optional imports for modules that may not exist yet
try:
    from .predictive_maintenance import PredictiveMaintenance
except ImportError:
    PredictiveMaintenance = None

try:
    from .optimization_engine import OptimizationEngine
except ImportError:
    OptimizationEngine = None

__all__ = [
    "PredictiveNetworkAnalyzer",
    "create_network_analyzer",
    "NetworkAnomalyDetector",
    "CriticalFailureDetector",
    "create_anomaly_detector",
    "create_critical_failure_detector",
    "PredictiveMaintenance",
    "OptimizationEngine"
] 