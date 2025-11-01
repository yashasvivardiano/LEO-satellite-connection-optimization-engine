#!/usr/bin/env python3
"""
Streamlit Cloud Entry Point
This file is used by Streamlit Cloud to run the dashboard.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import directly from dashboard module to avoid __init__.py import issues
import importlib.util
dashboard_path = project_root / "src" / "monitoring" / "dashboard.py"
spec = importlib.util.spec_from_file_location("dashboard", dashboard_path)
dashboard_module = importlib.util.module_from_spec(spec)
sys.modules["dashboard"] = dashboard_module
spec.loader.exec_module(dashboard_module)

# Run the dashboard
if __name__ == "__main__":
    dashboard_module.main()

