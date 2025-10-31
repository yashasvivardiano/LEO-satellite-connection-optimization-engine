#!/usr/bin/env python3
"""
Launch script for AI Network Stabilization Dashboard
Industry-standard project structure launcher
"""

import os
import sys
import webbrowser
import time
import threading
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    try:
        import flask
        import flask_cors
        print("SUCCESS: Backend dependencies available")
        return True
    except ImportError:
        print("Installing backend dependencies...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 
                'flask', 'flask-cors', 'tensorflow', 'scikit-learn', 'pandas', 'numpy'
            ])
            print("SUCCESS: Dependencies installed successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to install dependencies: {e}")
            return False

def start_backend():
    """Start the Flask backend server"""
    print("Starting backend server...")
    
    backend_dir = Path(__file__).parent / "backend"
    os.chdir(backend_dir)
    
    try:
        # Start Flask server
        subprocess.Popen([
            sys.executable, "server.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("SUCCESS: Backend server started on http://localhost:5000")
        return True
    except Exception as e:
        print(f"ERROR: Failed to start backend: {e}")
        return False

def open_dashboard():
    """Open the dashboard in the default browser"""
    print("Opening dashboard...")
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        webbrowser.open("http://localhost:5000")
        print("SUCCESS: Dashboard opened in browser")
        return True
    except Exception as e:
        print(f"ERROR: Failed to open dashboard: {e}")
        return False

def main():
    """Main launch function"""
    print("=" * 60)
    print("AI Network Stabilization Dashboard Launcher")
    print("=" * 60)
    print()
    
    # Check project structure
    print("Checking project structure...")
    frontend_dir = Path(__file__).parent / "frontend"
    backend_dir = Path(__file__).parent / "backend"
    
    if not frontend_dir.exists():
        print("ERROR: Frontend directory not found")
        return False
    
    if not backend_dir.exists():
        print("ERROR: Backend directory not found")
        return False
    
    print("SUCCESS: Project structure looks good")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("ERROR: Dependency check failed")
        return False
    print()
    
    # Start backend
    if not start_backend():
        print("ERROR: Failed to start backend")
        return False
    print()
    
    # Open dashboard
    if not open_dashboard():
        print("ERROR: Failed to open dashboard")
        return False
    print()
    
    print("=" * 60)
    print("SUCCESS: Dashboard launched successfully!")
    print("=" * 60)
    print()
    print("Dashboard URL: http://localhost:5000")
    print("To stop: Press Ctrl+C")
    print("Project structure:")
    print("   ├── frontend/     # HTML/CSS/JS")
    print("   ├── backend/      # Python/Flask API")
    print("   └── src/          # Core AI/Simulation")
    print()
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        print("SUCCESS: Dashboard stopped")

if __name__ == "__main__":
    main()
