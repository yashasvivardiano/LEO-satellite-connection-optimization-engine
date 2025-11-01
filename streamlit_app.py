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

# Import and run the dashboard
from src.monitoring.dashboard import main

if __name__ == "__main__":
    main()

