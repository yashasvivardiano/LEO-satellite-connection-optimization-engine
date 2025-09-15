#!/usr/bin/env python3
"""
MVP Dashboard Launcher

This script launches the AI Network Stabilization MVP dashboard.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the MVP dashboard."""
    print("🚀 Starting AI Network Stabilization MVP Dashboard...")
    print("=" * 60)
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✅ Streamlit is available")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit", "plotly"])
        print("✅ Streamlit installed")
    
    # Get the dashboard path
    dashboard_path = Path(__file__).parent / "src" / "monitoring" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard not found at {dashboard_path}")
        return 1
    
    print(f"📊 Launching dashboard: {dashboard_path}")
    print("\n🌐 The dashboard will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
