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
    
    # Try to find an available port
    import socket
    
    def is_port_available(port):
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    # Try ports starting from 8501
    port = 8501
    max_attempts = 10
    for attempt in range(max_attempts):
        if is_port_available(port):
            break
        print(f"⚠️  Port {port} is in use, trying {port + 1}...")
        port += 1
    else:
        print(f"❌ Could not find an available port after {max_attempts} attempts")
        return 1
    
    print(f"📊 Launching dashboard: {dashboard_path}")
    print(f"\n🌐 The dashboard will open in your browser at: http://localhost:{port}")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("=" * 60)
    
    try:
        # Launch streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", str(port),
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
