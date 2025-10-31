#!/usr/bin/env python3
"""
MVP Streamlit Dashboard

This dashboard demonstrates the AI-Based Network Stabilization Tool MVP
with real-time hybrid network simulation, AI-driven path switching,
and anomaly detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import sys
from pathlib import Path
import math

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.simulation.satellite_link import create_simulator
from src.ai.anomaly_detector import create_anomaly_detector, create_critical_failure_detector
from src.ai.network_analyzer import create_network_analyzer

# Page configuration
st.set_page_config(
    page_title="AI Network Stabilization MVP",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-critical {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #c62828;
    }
    .alert-warning {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef6c00;
    }
    .alert-success {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2e7d32;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'simulator' not in st.session_state:
        st.session_state.simulator = create_simulator()
    
    if 'anomaly_detector' not in st.session_state:
        st.session_state.anomaly_detector = create_anomaly_detector()
    
    if 'critical_detector' not in st.session_state:
        st.session_state.critical_detector = create_critical_failure_detector()
    
    if 'network_analyzer' not in st.session_state:
        st.session_state.network_analyzer = create_network_analyzer()
    
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = []
    
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    if 'auto_switch' not in st.session_state:
        st.session_state.auto_switch = True

def generate_network_data():
    """Generate new network data and run AI analysis."""
    simulator = st.session_state.simulator
    anomaly_detector = st.session_state.anomaly_detector
    critical_detector = st.session_state.critical_detector
    network_analyzer = st.session_state.network_analyzer
    
    # Generate telemetry
    telemetry = simulator.generate_telemetry()
    network_data = simulator.get_simulation_data()
    events = simulator.get_event_summary()
    
    # Run AI analysis
    anomaly_result = anomaly_detector.detect_anomaly(network_data)
    prediction = network_analyzer.predict_optimal_path(network_data)
    switch_decision = network_analyzer.should_switch_path(
        simulator.active_path, network_data
    )
    
    # Check for critical failures
    wired_critical = critical_detector.detect_critical_failure(network_data, "wired")
    satellite_critical = critical_detector.detect_critical_failure(network_data, "satellite")
    
    # Auto-switch if enabled and recommended
    if st.session_state.auto_switch and switch_decision.get('should_switch', False):
        new_path = switch_decision.get('predicted_path', simulator.active_path)
        simulator.switch_path(new_path)
        st.session_state.alerts.append({
            'timestamp': datetime.now(),
            'type': 'switch',
            'message': f"🔄 Proactive Switch: {simulator.active_path} → {new_path}",
            'severity': 'info'
        })
    
    # Check for anomalies and critical failures
    if anomaly_result.get('is_anomaly', False):
        st.session_state.alerts.append({
            'timestamp': datetime.now(),
            'type': 'anomaly',
            'message': f"⚠️ Anomaly detected! Error: {anomaly_result.get('reconstruction_error', 0):.4f}",
            'severity': 'warning'
        })
    
    if wired_critical.get('is_critical', False):
        st.session_state.alerts.append({
            'timestamp': datetime.now(),
            'type': 'critical',
            'message': f"🚨 CRITICAL: Wired path failure - {', '.join(wired_critical.get('failures', []))}",
            'severity': 'critical'
        })
    
    if satellite_critical.get('is_critical', False):
        st.session_state.alerts.append({
            'timestamp': datetime.now(),
            'type': 'critical',
            'message': f"🚨 CRITICAL: Satellite path failure - {', '.join(satellite_critical.get('failures', []))}",
            'severity': 'critical'
        })
    
    # Store data
    data_point = {
        'timestamp': datetime.now(),
        'wired_latency': telemetry['wired'].latency_ms,
        'wired_jitter': telemetry['wired'].jitter_ms,
        'wired_loss': telemetry['wired'].packet_loss_pct,
        'wired_bandwidth': telemetry['wired'].bandwidth_mbps,
        'wired_cost': telemetry['wired'].quality_cost,
        'satellite_latency': telemetry['satellite'].latency_ms,
        'satellite_jitter': telemetry['satellite'].jitter_ms,
        'satellite_loss': telemetry['satellite'].packet_loss_pct,
        'satellite_bandwidth': telemetry['satellite'].bandwidth_mbps,
        'satellite_cost': telemetry['satellite'].quality_cost,
        'active_path': simulator.active_path,
        'optimal_path': network_data['current_optimal_path'],
        'predicted_path': prediction.get('predicted_path', 'N/A'),
        'prediction_confidence': prediction.get('confidence', 0),
        'anomaly_detected': anomaly_result.get('is_anomaly', False),
        'anomaly_confidence': anomaly_result.get('confidence', 0),
        'active_events': len(events['active_events'])
    }
    
    st.session_state.simulation_data.append(data_point)
    
    # Keep only last 100 data points
    if len(st.session_state.simulation_data) > 100:
        st.session_state.simulation_data = st.session_state.simulation_data[-100:]
    
    return data_point, events

def create_metrics_display(data_point):
    """Create metrics display cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Active Path",
            data_point['active_path'].upper(),
            delta="Optimal" if data_point['active_path'] == data_point['optimal_path'] else "Suboptimal"
        )
    
    with col2:
        st.metric(
            "Wired Latency",
            f"{data_point['wired_latency']:.1f} ms",
            delta=f"{data_point['wired_jitter']:.1f} jitter"
        )
    
    with col3:
        st.metric(
            "Satellite Latency", 
            f"{data_point['satellite_latency']:.1f} ms",
            delta=f"{data_point['satellite_jitter']:.1f} jitter"
        )
    
    with col4:
        st.metric(
            "AI Confidence",
            f"{data_point['prediction_confidence']:.1%}",
            delta="Anomaly" if data_point['anomaly_detected'] else "Normal"
        )

def create_network_charts(data_df):
    """Create network performance charts."""
    if data_df.empty:
        return
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Latency Comparison', 'Packet Loss', 'Bandwidth', 'Quality Cost'),
        vertical_spacing=0.1
    )
    
    # Latency comparison
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['wired_latency'], 
                  name='Wired Latency', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['satellite_latency'], 
                  name='Satellite Latency', line=dict(color='red')),
        row=1, col=1
    )
    
    # Packet loss
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['wired_loss'], 
                  name='Wired Loss', line=dict(color='blue')),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['satellite_loss'], 
                  name='Satellite Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # Bandwidth
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['wired_bandwidth'], 
                  name='Wired Bandwidth', line=dict(color='blue')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['satellite_bandwidth'], 
                  name='Satellite Bandwidth', line=dict(color='red')),
        row=2, col=1
    )
    
    # Quality cost
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['wired_cost'], 
                  name='Wired Cost', line=dict(color='blue')),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=data_df['timestamp'], y=data_df['satellite_cost'], 
                  name='Satellite Cost', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

def create_path_switching_chart(data_df):
    """Create path switching visualization."""
    if data_df.empty:
        return
    
    # Create path switching timeline
    fig = go.Figure()
    
    # Add path indicators
    for i, row in data_df.iterrows():
        color = 'blue' if row['active_path'] == 'wired' else 'red'
        fig.add_trace(go.Scatter(
            x=[row['timestamp']],
            y=[1 if row['active_path'] == 'wired' else 0],
            mode='markers',
            marker=dict(size=10, color=color),
            name=row['active_path'],
            showlegend=False
        ))
    
    fig.update_layout(
        title="Path Switching Timeline",
        xaxis_title="Time",
        yaxis_title="Path (0=Satellite, 1=Wired)",
        yaxis=dict(tickvals=[0, 1], ticktext=['Satellite', 'Wired']),
        height=200
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_3d_globe(data_point):
    """Create 3D globe visualization with satellites and ground stations."""
    # Ground stations (fixed positions)
    ground_stations = [
        {'name': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'quality': data_point['wired_cost']},
        {'name': 'London', 'lat': 51.5074, 'lon': -0.1278, 'quality': data_point['wired_cost']},
        {'name': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'quality': data_point['wired_cost']},
        {'name': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'quality': data_point['wired_cost']},
        {'name': 'São Paulo', 'lat': -23.5505, 'lon': -46.6333, 'quality': data_point['wired_cost']}
    ]
    
    # Satellite positions (simplified orbital simulation)
    satellites = []
    time_factor = datetime.now().timestamp() / 1000  # Slow down orbital motion
    
    for i in range(6):  # 6 satellites
        # Simple orbital simulation
        angle = (time_factor + i * 60) % (2 * math.pi)
        altitude = 550  # km
        lat = 60 * math.sin(angle)  # Inclined orbit
        lon = (angle * 180 / math.pi + i * 60) % 360 - 180
        
        # Quality based on current satellite metrics
        quality = data_point['satellite_cost']
        satellites.append({
            'name': f'Sat-{i+1}',
            'lat': lat,
            'lon': lon,
            'altitude': altitude,
            'quality': quality
        })
    
    # Create 3D globe
    fig = go.Figure()
    
    # Add Earth sphere
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=100, color='lightblue', opacity=0.3),
        name='Earth',
        showlegend=False
    ))
    
    # Add ground stations
    for station in ground_stations:
        # Convert lat/lon to 3D coordinates
        lat_rad = math.radians(station['lat'])
        lon_rad = math.radians(station['lon'])
        x = math.cos(lat_rad) * math.cos(lon_rad)
        y = math.cos(lat_rad) * math.sin(lon_rad)
        z = math.sin(lat_rad)
        
        # Color based on quality (red = bad, green = good)
        # Clamp values to valid RGB range (0-255)
        red = max(0, min(255, int(255 * station["quality"])))
        green = max(0, min(255, int(255 * (1 - station["quality"]))))
        quality_color = f'rgb({red}, {green}, 0)'
        
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=8, color=quality_color),
            text=[station['name']],
            textposition='top center',
            name=station['name'],
            showlegend=False
        ))
    
    # Add satellites
    for sat in satellites:
        # Convert lat/lon to 3D coordinates with altitude
        lat_rad = math.radians(sat['lat'])
        lon_rad = math.radians(sat['lon'])
        scale = 1 + sat['altitude'] / 6371  # Earth radius = 6371 km
        x = scale * math.cos(lat_rad) * math.cos(lon_rad)
        y = scale * math.cos(lat_rad) * math.sin(lon_rad)
        z = scale * math.sin(lat_rad)
        
        # Color based on quality
        # Clamp values to valid RGB range (0-255)
        red = max(0, min(255, int(255 * sat["quality"])))
        green = max(0, min(255, int(255 * (1 - sat["quality"]))))
        quality_color = f'rgb({red}, {green}, 0)'
        
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            marker=dict(size=6, color=quality_color, symbol='diamond'),
            text=[sat['name']],
            textposition='top center',
            name=sat['name'],
            showlegend=False
        ))
    
    # Add connections between ground stations (wired network)
    for i, station1 in enumerate(ground_stations):
        for j, station2 in enumerate(ground_stations[i+1:], i+1):
            lat1_rad = math.radians(station1['lat'])
            lon1_rad = math.radians(station1['lon'])
            lat2_rad = math.radians(station2['lat'])
            lon2_rad = math.radians(station2['lon'])
            
            x1 = math.cos(lat1_rad) * math.cos(lon1_rad)
            y1 = math.cos(lat1_rad) * math.sin(lon1_rad)
            z1 = math.sin(lat1_rad)
            
            x2 = math.cos(lat2_rad) * math.cos(lon2_rad)
            y2 = math.cos(lat2_rad) * math.sin(lon2_rad)
            z2 = math.sin(lat2_rad)
            
            # Average quality for connection color
            avg_quality = (station1['quality'] + station2['quality']) / 2
            red = max(0, min(255, int(255 * avg_quality)))
            green = max(0, min(255, int(255 * (1 - avg_quality))))
            connection_color = f'rgb({red}, {green}, 0)'
            
            fig.add_trace(go.Scatter3d(
                x=[x1, x2], y=[y1, y2], z=[z1, z2],
                mode='lines',
                line=dict(color=connection_color, width=3),
                name=f'Connection {station1["name"]}-{station2["name"]}',
                showlegend=False
            ))
    
    # Add satellite-to-ground connections (when satellite is overhead)
    for sat in satellites:
        for station in ground_stations:
            # Check if satellite is "overhead" (simplified: within 30 degrees)
            lat_diff = abs(sat['lat'] - station['lat'])
            lon_diff = abs(sat['lon'] - station['lon'])
            
            if lat_diff < 30 and lon_diff < 30:  # Simplified overhead check
                lat1_rad = math.radians(station['lat'])
                lon1_rad = math.radians(station['lon'])
                lat2_rad = math.radians(sat['lat'])
                lon2_rad = math.radians(sat['lon'])
                
                x1 = math.cos(lat1_rad) * math.cos(lon1_rad)
                y1 = math.cos(lat1_rad) * math.sin(lon1_rad)
                z1 = math.sin(lat1_rad)
                
                scale = 1 + sat['altitude'] / 6371
                x2 = scale * math.cos(lat2_rad) * math.cos(lon2_rad)
                y2 = scale * math.cos(lat2_rad) * math.sin(lon2_rad)
                z2 = scale * math.sin(lat2_rad)
                
                # Satellite connection color
                sat_quality = sat['quality']
                red = max(0, min(255, int(255 * sat_quality)))
                green = max(0, min(255, int(255 * (1 - sat_quality))))
                connection_color = f'rgb({red}, {green}, 0)'
                
                fig.add_trace(go.Scatter3d(
                    x=[x1, x2], y=[y1, y2], z=[z1, z2],
                    mode='lines',
                    line=dict(color=connection_color, width=2, dash='dot'),
                    name=f'Sat-{sat["name"]}-{station["name"]}',
                    showlegend=False
                ))
    
    # Update layout
    fig.update_layout(
        title="🛰️ Global Network Topology",
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            bgcolor='black',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        height=600,
        showlegend=False
    )
    
    return fig

def create_2d_network_visualization(data_point):
    """Create a simple 2D network visualization as fallback."""
    # Create a simple network diagram
    fig = go.Figure()
    
    # Ground stations
    ground_stations = [
        {'name': 'New York', 'x': 0, 'y': 0, 'quality': data_point['wired_cost']},
        {'name': 'London', 'x': 2, 'y': 1, 'quality': data_point['wired_cost']},
        {'name': 'Tokyo', 'x': 4, 'y': 0, 'quality': data_point['wired_cost']},
        {'name': 'Sydney', 'x': 2, 'y': -2, 'quality': data_point['wired_cost']}
    ]
    
    # Satellites
    satellites = [
        {'name': 'Sat-1', 'x': 1, 'y': 2, 'quality': data_point['satellite_cost']},
        {'name': 'Sat-2', 'x': 3, 'y': 2, 'quality': data_point['satellite_cost']},
        {'name': 'Sat-3', 'x': 1, 'y': -1, 'quality': data_point['satellite_cost']}
    ]
    
    # Add ground stations
    for station in ground_stations:
        color = 'green' if station['quality'] < 1.0 else 'red'
        fig.add_trace(go.Scatter(
            x=[station['x']], y=[station['y']],
            mode='markers+text',
            marker=dict(size=15, color=color, symbol='circle'),
            text=[station['name']],
            textposition='top center',
            name=station['name'],
            showlegend=False
        ))
    
    # Add satellites
    for sat in satellites:
        color = 'green' if sat['quality'] < 1.0 else 'red'
        fig.add_trace(go.Scatter(
            x=[sat['x']], y=[sat['y']],
            mode='markers+text',
            marker=dict(size=12, color=color, symbol='diamond'),
            text=[sat['name']],
            textposition='top center',
            name=sat['name'],
            showlegend=False
        ))
    
    # Add connections between ground stations
    for i, station1 in enumerate(ground_stations):
        for j, station2 in enumerate(ground_stations[i+1:], i+1):
            fig.add_trace(go.Scatter(
                x=[station1['x'], station2['x']], 
                y=[station1['y'], station2['y']],
                mode='lines',
                line=dict(color='blue', width=2),
                showlegend=False
            ))
    
    # Add satellite connections (simplified)
    for sat in satellites:
        for station in ground_stations:
            fig.add_trace(go.Scatter(
                x=[station['x'], sat['x']], 
                y=[station['y'], sat['y']],
                mode='lines',
                line=dict(color='orange', width=1, dash='dot'),
                showlegend=False
            ))
    
    fig.update_layout(
        title="2D Network Topology (Fallback)",
        xaxis=dict(showticklabels=False, showgrid=True),
        yaxis=dict(showticklabels=False, showgrid=True),
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_alerts():
    """Display recent alerts."""
    if not st.session_state.alerts:
        st.info("No alerts in the last 10 minutes")
        return
    
    # Show last 10 alerts
    recent_alerts = st.session_state.alerts[-10:]
    
    for alert in reversed(recent_alerts):
        alert_class = f"alert-{alert['severity']}"
        st.markdown(f"""
        <div class="{alert_class}">
            <strong>{alert['timestamp'].strftime('%H:%M:%S')}</strong> - {alert['message']}
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main dashboard function."""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🛰️ AI Network Stabilization MVP</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Simulation controls
        st.subheader("Simulation")
        if st.button("▶️ Start Simulation", disabled=st.session_state.is_running, key="start_btn"):
            st.session_state.is_running = True
            st.success("✅ Starting simulation...")
            st.rerun()
        
        if st.button("⏸️ Stop Simulation", disabled=not st.session_state.is_running, key="stop_btn"):
            st.session_state.is_running = False
            st.rerun()
        
        if st.button("🔄 Reset Simulation", key="reset_btn"):
            st.session_state.simulation_data = []
            st.session_state.alerts = []
            st.session_state.simulator = create_simulator()
            st.session_state.is_running = False
            st.rerun()
        
        # Refresh button (only shown when simulation is running)
        if st.session_state.is_running:
            st.divider()
            if st.button("🔄 Refresh Data", help="Click to update simulation data", key="refresh_btn"):
                st.rerun()
        
        # AI controls
        st.subheader("AI Settings")
        auto_switch_new = st.checkbox("Auto Path Switching", value=st.session_state.auto_switch)
        # Only update and rerun if checkbox actually changed
        if auto_switch_new != st.session_state.auto_switch:
            st.session_state.auto_switch = auto_switch_new
        
        # Manual path switching
        st.subheader("Manual Control")
        if st.button("Switch to Wired", key="switch_wired_btn"):
            if st.session_state.is_running:
                st.session_state.simulator.switch_path("wired")
                st.rerun()
        
        if st.button("Switch to Satellite", key="switch_satellite_btn"):
            if st.session_state.is_running:
                st.session_state.simulator.switch_path("satellite")
                st.rerun()
        
        # Statistics
        st.subheader("Statistics")
        if st.session_state.simulation_data:
            data_df = pd.DataFrame(st.session_state.simulation_data)
            st.metric("Data Points", len(data_df))
            st.metric("Alerts", len(st.session_state.alerts))
            st.metric("Path Switches", len([a for a in st.session_state.alerts if a['type'] == 'switch']))
    
    # Main content
    if st.session_state.is_running:
        # Create placeholders to prevent page jump
        metrics_placeholder = st.empty()
        charts_placeholder = st.empty()
        events_placeholder = st.empty()
        globe_placeholder = st.empty()
        status_placeholder = st.empty()
        
        try:
            # Generate new data
            with status_placeholder.container():
                st.info("🔄 Generating network data and running AI analysis...")
            
            try:
                # Generate data (AI predictions may take a moment)
                data_point, events = generate_network_data()
                status_placeholder.empty()  # Clear status once data is generated
            except Exception as e:
                status_placeholder.empty()
                with metrics_placeholder.container():
                    st.error(f"❌ Error generating network data: {str(e)}")
                    import traceback
                    with st.expander("🔍 Show Error Details"):
                        st.code(traceback.format_exc())
                    st.warning("💡 Try clicking 'Reset Simulation' to restart")
                # Show error in status too
                status_placeholder.error("❌ Failed to generate data - see error above")
                return  # Stop here, don't try to display anything else
            
            # Display metrics in placeholder
            with metrics_placeholder.container():
                try:
                    # Debug: Show we have data
                    if data_point:
                        create_metrics_display(data_point)
                    else:
                        st.error("⚠️ No data point generated!")
                except Exception as e:
                    st.error(f"Error displaying metrics: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    # Try to show raw data if metrics fail
                    st.write("**Debug - Raw data_point:**", data_point if 'data_point' in locals() else "Not available")
            
            # Display 3D globe (lazy load - only show if explicitly requested or first time)
            show_globe = st.session_state.get('show_globe', True)
            with globe_placeholder.container():
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader("🌍 Global Network Topology")
                with col2:
                    show_globe = st.checkbox("Show 3D Globe", value=show_globe, key="globe_toggle")
                    st.session_state.show_globe = show_globe
                
                if show_globe:
                    try:
                        with st.spinner("Generating 3D visualization..."):
                            globe_fig = create_3d_globe(data_point)
                            st.plotly_chart(globe_fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"3D visualization issue: {str(e)}. Using 2D visualization...")
                        create_2d_network_visualization(data_point)
                else:
                    # Always show 2D when 3D is disabled
                    create_2d_network_visualization(data_point)
            
            # Display charts in placeholder
            if st.session_state.simulation_data:
                data_df = pd.DataFrame(st.session_state.simulation_data)
                with charts_placeholder.container():
                    st.subheader("📊 Network Performance Charts")
                    create_network_charts(data_df)
                    
                    # Add path switching timeline
                    st.subheader("🔄 Path Switching Timeline")
                    create_path_switching_chart(data_df)
            
            # Display events in placeholder
            if events and events.get('active_events'):
                with events_placeholder.container():
                    st.subheader("⚡ Active Events")
                    for event in events['active_events']:
                        st.warning(f"**{event['type'].upper()}**: {event['description']} (Remaining: {event['remaining_seconds']:.0f}s)")
            
            # Show status
            with status_placeholder.container():
                st.success("✅ Simulation data generated successfully!")
                st.info("💡 **Tip:** Disable 'Show 3D Globe' to speed up refreshes. AI analysis takes a few seconds.")
        
        except Exception as e:
            with metrics_placeholder.container():
                st.error(f"❌ Unexpected error in simulation: {e}")
                import traceback
                with st.expander("🔍 Show Error Details"):
                    st.code(traceback.format_exc())
                st.warning("💡 Try clicking 'Reset Simulation' to restart")
    
    else:
        # Welcome screen
        st.info("👆 Click 'Start Simulation' to begin the AI Network Stabilization demonstration")
        
        # Show recent data if available
        if st.session_state.simulation_data:
            data_df = pd.DataFrame(st.session_state.simulation_data)
            st.subheader("📊 Last Simulation Data")
            create_network_charts(data_df)
    
    # Alerts section
    st.subheader("🚨 Alerts & Notifications")
    display_alerts()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>AI-Based Network Stabilization Tool MVP | Week 7-8 Implementation</p>
        <p>Demonstrating dual-AI strategy: Proactive optimization + Reactive failure detection</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()