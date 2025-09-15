#!/usr/bin/env python3
"""
Test script to verify the new industry-standard project structure
"""

import os
import sys
from pathlib import Path

def test_structure():
    """Test the project structure"""
    print("🔍 Testing Industry-Standard Project Structure")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Frontend structure
    print("\n1. Testing Frontend Structure...")
    frontend_dir = base_dir / "frontend"
    if frontend_dir.exists():
        total_tests += 1
        if (frontend_dir / "index.html").exists():
            print("   ✅ index.html exists")
            tests_passed += 1
        else:
            print("   ❌ index.html missing")
        
        total_tests += 1
        if (frontend_dir / "assets" / "css" / "dashboard.css").exists():
            print("   ✅ CSS file exists")
            tests_passed += 1
        else:
            print("   ❌ CSS file missing")
        
        total_tests += 1
        if (frontend_dir / "assets" / "js" / "dashboard.js").exists():
            print("   ✅ JavaScript file exists")
            tests_passed += 1
        else:
            print("   ❌ JavaScript file missing")
    else:
        print("   ❌ Frontend directory missing")
        total_tests += 3
    
    # Test 2: Backend structure
    print("\n2. Testing Backend Structure...")
    backend_dir = base_dir / "backend"
    if backend_dir.exists():
        total_tests += 1
        if (backend_dir / "server.py").exists():
            print("   ✅ server.py exists")
            tests_passed += 1
        else:
            print("   ❌ server.py missing")
        
        total_tests += 1
        if (backend_dir / "config.py").exists():
            print("   ✅ config.py exists")
            tests_passed += 1
        else:
            print("   ❌ config.py missing")
        
        total_tests += 1
        if (backend_dir / "requirements.txt").exists():
            print("   ✅ requirements.txt exists")
            tests_passed += 1
        else:
            print("   ❌ requirements.txt missing")
    else:
        print("   ❌ Backend directory missing")
        total_tests += 3
    
    # Test 3: Documentation
    print("\n3. Testing Documentation...")
    total_tests += 1
    if (base_dir / "PROJECT_STRUCTURE.md").exists():
        print("   ✅ PROJECT_STRUCTURE.md exists")
        tests_passed += 1
    else:
        print("   ❌ PROJECT_STRUCTURE.md missing")
    
    total_tests += 1
    if (frontend_dir / "README.md").exists():
        print("   ✅ Frontend README exists")
        tests_passed += 1
    else:
        print("   ❌ Frontend README missing")
    
    total_tests += 1
    if (backend_dir / "README.md").exists():
        print("   ✅ Backend README exists")
        tests_passed += 1
    else:
        print("   ❌ Backend README missing")
    
    # Test 4: Launch script
    print("\n4. Testing Launch Script...")
    total_tests += 1
    if (base_dir / "launch_dashboard.py").exists():
        print("   ✅ launch_dashboard.py exists")
        tests_passed += 1
    else:
        print("   ❌ launch_dashboard.py missing")
    
    # Results
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Structure is industry-standard.")
        print("\n🚀 Ready to launch:")
        print("   python launch_dashboard.py")
        return True
    else:
        print("❌ Some tests failed. Check the structure.")
        return False

if __name__ == "__main__":
    test_structure()
