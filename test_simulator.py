#!/usr/bin/env python3
"""Test script to debug the simulator."""

import sys
sys.path.append('.')

from src.simulation.satellite_link import create_simulator
from src.ai.anomaly_detector import create_anomaly_detector
from src.ai.network_analyzer import create_network_analyzer

print("🧪 Testing simulator components...")

try:
    print("1. Creating simulator...")
    simulator = create_simulator()
    print(f"   ✅ Simulator created, active_path: {simulator.active_path}")
    
    print("2. Generating telemetry...")
    telemetry = simulator.generate_telemetry()
    print(f"   ✅ Wired latency: {telemetry['wired'].latency_ms} ms")
    print(f"   ✅ Satellite latency: {telemetry['satellite'].latency_ms} ms")
    
    print("3. Getting simulation data...")
    network_data = simulator.get_simulation_data()
    print(f"   ✅ Network data keys: {list(network_data.keys())[:5]}...")
    
    print("4. Testing anomaly detector...")
    detector = create_anomaly_detector()
    anomaly_result = detector.detect_anomaly(network_data)
    print(f"   ✅ Anomaly detected: {anomaly_result.get('is_anomaly', False)}")
    
    print("5. Testing network analyzer...")
    analyzer = create_network_analyzer()
    prediction = analyzer.predict_optimal_path(network_data)
    print(f"   ✅ Predicted path: {prediction.get('predicted_path', 'N/A')}")
    
    print("\n🎉 All components working!")
    
except Exception as e:
    print(f"\n❌ Error occurred: {e}")
    import traceback
    traceback.print_exc()

