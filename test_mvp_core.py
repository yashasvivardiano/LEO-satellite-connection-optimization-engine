#!/usr/bin/env python3
"""
MVP Core Components Test Script

This script tests the core MVP components:
1. Hybrid Network Simulator
2. AI Anomaly Detector  
3. Predictive Network Analyzer

Run this to verify everything is working before building the dashboard.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.simulation.satellite_link import create_simulator
from src.ai.anomaly_detector import create_anomaly_detector, create_critical_failure_detector
from src.ai.network_analyzer import create_network_analyzer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hybrid_simulator():
    """Test the hybrid network simulator."""
    print("\n🛰️ Testing Hybrid Network Simulator")
    print("=" * 50)
    
    try:
        simulator = create_simulator()
        
        # Generate some telemetry
        for i in range(5):
            telemetry = simulator.generate_telemetry()
            data = simulator.get_simulation_data()
            events = simulator.get_event_summary()
            
            print(f"\nStep {i+1}:")
            print(f"  Wired: {telemetry['wired'].latency_ms:.1f}ms, "
                  f"{telemetry['wired'].quality_cost:.3f} cost")
            print(f"  Satellite: {telemetry['satellite'].latency_ms:.1f}ms, "
                  f"{telemetry['satellite'].quality_cost:.3f} cost")
            print(f"  Optimal: {data['current_optimal_path']}")
            print(f"  Active: {data['active_path']}")
            
            if events['active_events']:
                print(f"  Events: {len(events['active_events'])} active")
        
        print("✅ Hybrid Network Simulator: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Hybrid Network Simulator: FAILED - {e}")
        return False

def test_anomaly_detector():
    """Test the anomaly detector."""
    print("\n🔍 Testing Anomaly Detector")
    print("=" * 50)
    
    try:
        detector = create_anomaly_detector()
        
        # Test with sample data
        sample_data = {
            "wired_latency_ms": 30.0,
            "wired_jitter_ms": 6.0,
            "wired_packet_loss_pct": 0.015,
            "wired_bandwidth_mbps": 100.0,
            "wired_quality_cost": 0.8,
            "satellite_latency_ms": 45.0,
            "satellite_jitter_ms": 8.0,
            "satellite_packet_loss_pct": 0.05,
            "satellite_bandwidth_mbps": 80.0,
            "satellite_quality_cost": 0.6
        }
        
        # Test anomaly detection
        result = detector.detect_anomaly(sample_data)
        print(f"Anomaly detection result: {result}")
        
        # Test critical failure detector
        critical_detector = create_critical_failure_detector()
        critical_result = critical_detector.detect_critical_failure(sample_data, "wired")
        print(f"Critical failure detection result: {critical_result}")
        
        print("✅ Anomaly Detector: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Anomaly Detector: FAILED - {e}")
        return False

def test_network_analyzer():
    """Test the network analyzer."""
    print("\n🧠 Testing Network Analyzer")
    print("=" * 50)
    
    try:
        analyzer = create_network_analyzer()
        
        # Test with sample data
        sample_data = {
            "wired_latency_ms": 30.0,
            "wired_jitter_ms": 6.0,
            "wired_packet_loss_pct": 0.015,
            "wired_bandwidth_mbps": 100.0,
            "wired_quality_cost": 0.8,
            "satellite_latency_ms": 45.0,
            "satellite_jitter_ms": 8.0,
            "satellite_packet_loss_pct": 0.05,
            "satellite_bandwidth_mbps": 80.0,
            "satellite_quality_cost": 0.6,
            "active_path": "wired"
        }
        
        # Test path prediction
        prediction = analyzer.predict_optimal_path(sample_data)
        print(f"Path prediction: {prediction}")
        
        # Test switch decision
        switch_decision = analyzer.should_switch_path("wired", sample_data)
        print(f"Switch decision: {switch_decision}")
        
        # Test network insights
        insights = analyzer.get_network_insights(sample_data)
        print(f"Network insights: {insights}")
        
        print("✅ Network Analyzer: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Network Analyzer: FAILED - {e}")
        return False

def test_integrated_system():
    """Test the integrated system."""
    print("\n🔗 Testing Integrated System")
    print("=" * 50)
    
    try:
        # Create all components
        simulator = create_simulator()
        anomaly_detector = create_anomaly_detector()
        critical_detector = create_critical_failure_detector()
        analyzer = create_network_analyzer()
        
        print("Running integrated simulation...")
        
        for i in range(10):
            # Generate network data
            telemetry = simulator.generate_telemetry()
            network_data = simulator.get_simulation_data()
            
            # Run AI analysis
            anomaly_result = anomaly_detector.detect_anomaly(network_data)
            prediction = analyzer.predict_optimal_path(network_data)
            switch_decision = analyzer.should_switch_path(
                simulator.active_path, network_data
            )
            
            # Check for critical failures
            wired_critical = critical_detector.detect_critical_failure(network_data, "wired")
            satellite_critical = critical_detector.detect_critical_failure(network_data, "satellite")
            
            # Log results
            if i % 3 == 0:  # Show every 3rd step
                print(f"\nStep {i+1}:")
                print(f"  Active Path: {simulator.active_path}")
                print(f"  Predicted Path: {prediction.get('predicted_path', 'N/A')}")
                print(f"  Switch Recommended: {switch_decision.get('should_switch', False)}")
                print(f"  Anomaly Detected: {anomaly_result.get('is_anomaly', False)}")
                print(f"  Wired Critical: {wired_critical.get('is_critical', False)}")
                print(f"  Satellite Critical: {satellite_critical.get('is_critical', False)}")
            
            # Simulate path switching if recommended
            if switch_decision.get('should_switch', False):
                new_path = switch_decision.get('predicted_path', simulator.active_path)
                simulator.switch_path(new_path)
        
        print("✅ Integrated System: PASSED")
        return True
        
    except Exception as e:
        print(f"❌ Integrated System: FAILED - {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 MVP Core Components Test Suite")
    print("=" * 60)
    
    tests = [
        ("Hybrid Network Simulator", test_hybrid_simulator),
        ("Anomaly Detector", test_anomaly_detector),
        ("Network Analyzer", test_network_analyzer),
        ("Integrated System", test_integrated_system)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: CRASHED - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! MVP core components are ready.")
        print("Next step: Create the Streamlit dashboard for demonstration.")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
