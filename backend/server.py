#!/usr/bin/env python3
"""
AI Network Stabilization - Investor Dashboard Server
Professional HTML/CSS/JS dashboard with real-time data from Python backend
"""

import json
import time
import random
from datetime import datetime
from flask import Flask, jsonify, render_template_string
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

# Global simulation state
simulation_data = {
    'is_running': False,
    'current_metrics': {
        'active_path': 'WIRED',
        'wired_latency': 32.8,
        'satellite_latency': 43.9,
        'wired_loss': 0.015,
        'satellite_loss': 0.05,
        'wired_bandwidth': 950,
        'satellite_bandwidth': 800,
        'wired_cost': 0.8,
        'satellite_cost': 1.2,
        'ai_confidence': 60.0
    },
    'history': [],
    'alerts': []
}

def generate_realistic_metrics():
    """Generate realistic network metrics with some variation"""
    base_time = datetime.now()
    
    # Simulate realistic network behavior
    wired_latency = 25 + random.uniform(0, 15) + random.uniform(-5, 5)
    satellite_latency = 40 + random.uniform(0, 20) + random.uniform(-8, 8)
    
    # Add occasional spikes
    if random.random() < 0.1:  # 10% chance of spike
        wired_latency += random.uniform(20, 50)
    if random.random() < 0.15:  # 15% chance of satellite spike
        satellite_latency += random.uniform(30, 80)
    
    wired_loss = max(0.001, 0.01 + random.uniform(0, 0.02))
    satellite_loss = max(0.01, 0.03 + random.uniform(0, 0.04))
    
    wired_bandwidth = 950 + random.uniform(-50, 50)
    satellite_bandwidth = 800 + random.uniform(-100, 100)
    
    # Calculate quality costs
    wired_cost = (wired_latency / 30) + (wired_loss * 100) + (1000 - wired_bandwidth) / 1000
    satellite_cost = (satellite_latency / 40) + (satellite_loss * 50) + (1000 - satellite_bandwidth) / 1000
    
    # Determine optimal path
    optimal_path = 'WIRED' if wired_cost < satellite_cost else 'SATELLITE'
    
    # AI confidence based on cost difference
    cost_diff = abs(wired_cost - satellite_cost)
    ai_confidence = min(95, max(60, 80 - cost_diff * 20))
    
    return {
        'timestamp': base_time.isoformat(),
        'active_path': optimal_path,
        'wired_latency': round(wired_latency, 1),
        'satellite_latency': round(satellite_latency, 1),
        'wired_loss': round(wired_loss * 100, 2),
        'satellite_loss': round(satellite_loss * 100, 2),
        'wired_bandwidth': round(wired_bandwidth, 0),
        'satellite_bandwidth': round(satellite_bandwidth, 0),
        'wired_cost': round(wired_cost, 2),
        'satellite_cost': round(satellite_cost, 2),
        'ai_confidence': round(ai_confidence, 1)
    }

def simulation_worker():
    """Background worker that generates simulation data"""
    while True:
        if simulation_data['is_running']:
            metrics = generate_realistic_metrics()
            simulation_data['current_metrics'] = metrics
            
            # Add to history (keep last 50 points)
            simulation_data['history'].append(metrics)
            if len(simulation_data['history']) > 50:
                simulation_data['history'].pop(0)
            
            # Generate occasional alerts
            if random.random() < 0.05:  # 5% chance of alert
                alert_types = [
                    ('warning', 'Satellite Handoff', 'Satellite handoff detected - switching to optimal path'),
                    ('success', 'Path Optimized', 'AI successfully optimized network path selection'),
                    ('warning', 'Congestion Detected', 'Network congestion detected on primary path'),
                    ('success', 'Anomaly Resolved', 'Network anomaly automatically resolved')
                ]
                alert_type, title, message = random.choice(alert_types)
                simulation_data['alerts'].append({
                    'type': alert_type,
                    'title': title,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Keep only last 10 alerts
                if len(simulation_data['alerts']) > 10:
                    simulation_data['alerts'].pop(0)
        
        time.sleep(3)  # Update every 3 seconds for better performance

@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    with open('../frontend/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/api/metrics')
def get_metrics():
    """Get current network metrics"""
    return jsonify(simulation_data['current_metrics'])

@app.route('/api/history')
def get_history():
    """Get historical data for charts"""
    return jsonify(simulation_data['history'])

@app.route('/api/alerts')
def get_alerts():
    """Get recent alerts"""
    return jsonify(simulation_data['alerts'])

@app.route('/api/start', methods=['POST'])
def start_simulation():
    """Start the simulation"""
    simulation_data['is_running'] = True
    simulation_data['alerts'].append({
        'type': 'success',
        'title': 'Simulation Started',
        'message': 'AI Network Stabilization is now monitoring the hybrid network',
        'timestamp': datetime.now().isoformat()
    })
    return jsonify({'status': 'started'})

@app.route('/api/stop', methods=['POST'])
def stop_simulation():
    """Stop the simulation"""
    simulation_data['is_running'] = False
    simulation_data['alerts'].append({
        'type': 'warning',
        'title': 'Simulation Stopped',
        'message': 'AI monitoring has been paused',
        'timestamp': datetime.now().isoformat()
    })
    return jsonify({'status': 'stopped'})

@app.route('/api/reset', methods=['POST'])
def reset_simulation():
    """Reset the simulation"""
    simulation_data['is_running'] = False
    simulation_data['history'] = []
    simulation_data['alerts'] = []
    simulation_data['current_metrics'] = {
        'active_path': 'WIRED',
        'wired_latency': 32.8,
        'satellite_latency': 43.9,
        'wired_loss': 0.015,
        'satellite_loss': 0.05,
        'wired_bandwidth': 950,
        'satellite_bandwidth': 800,
        'wired_cost': 0.8,
        'satellite_cost': 1.2,
        'ai_confidence': 60.0
    }
    return jsonify({'status': 'reset'})

@app.route('/api/switch/wired', methods=['POST'])
def switch_to_wired():
    """Switch to wired path"""
    simulation_data['current_metrics']['active_path'] = 'WIRED'
    simulation_data['alerts'].append({
        'type': 'success',
        'title': 'Path Switch',
        'message': 'Traffic switched to wired network',
        'timestamp': datetime.now().isoformat()
    })
    return jsonify({'status': 'switched', 'path': 'WIRED'})

@app.route('/api/switch/satellite', methods=['POST'])
def switch_to_satellite():
    """Switch to satellite path"""
    simulation_data['current_metrics']['active_path'] = 'SATELLITE'
    simulation_data['alerts'].append({
        'type': 'success',
        'title': 'Path Switch',
        'message': 'Traffic switched to satellite network',
        'timestamp': datetime.now().isoformat()
    })
    return jsonify({'status': 'switched', 'path': 'SATELLITE'})

if __name__ == '__main__':
    # Start background simulation worker
    simulation_thread = threading.Thread(target=simulation_worker, daemon=True)
    simulation_thread.start()
    
    print("🚀 AI Network Stabilization - Investor Dashboard Server")
    print("=" * 60)
    print("📊 Professional HTML/CSS/JS Dashboard")
    print("🌐 Open your browser to: http://localhost:5000")
    print("🎯 Perfect for investor presentations!")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=True)
