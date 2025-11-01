"""
Monitoring and observability components for satellite networks.

This module provides comprehensive monitoring, metrics collection,
and alerting systems for LEO satellite communication networks.
"""

# Optional imports - these modules are placeholders and may not have all classes yet
try:
    from .metrics_collector import MetricsCollector
except (ImportError, AttributeError):
    MetricsCollector = None

try:
    from .health_monitor import HealthMonitor
except (ImportError, AttributeError):
    HealthMonitor = None

try:
    from .alert_manager import AlertManager
except (ImportError, AttributeError):
    AlertManager = None

# Dashboard is the main module and should always be available
try:
    from .dashboard import Dashboard, main
except ImportError:
    Dashboard = None
    main = None

__all__ = [
    "MetricsCollector",
    "HealthMonitor", 
    "AlertManager",
    "Dashboard",
    "main"
] 