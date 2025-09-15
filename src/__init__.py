"""
LEO Satellite Connection Optimization Project

Advanced LEO satellite connection optimization and AI network analysis.
"""

__version__ = "0.2.0"
__author__ = "LEO Project Team"
__email__ = "team@leo-satellite.com"

# Avoid importing subpackages at module import time to prevent heavy side effects
# in lightweight contexts (e.g., Streamlit app startup). Subpackages can be
# imported explicitly where needed.

__all__ = ["core", "ai", "simulation", "monitoring", "utils"]