"""
Satellite link simulation tools.

This module provides comprehensive simulation capabilities for satellite
communication links, performance testing, and scenario modeling.
"""

from .satellite_link import HybridNetworkSimulator, create_simulator

# Optional imports for modules that may not exist yet
try:
    from .performance_tester import PerformanceTester
except ImportError:
    PerformanceTester = None

try:
    from .scenario_modeler import ScenarioModeler
except ImportError:
    ScenarioModeler = None

try:
    from .validation_tools import ValidationTools
except ImportError:
    ValidationTools = None

__all__ = [
    "HybridNetworkSimulator",
    "create_simulator",
    "PerformanceTester",
    "ScenarioModeler", 
    "ValidationTools"
] 