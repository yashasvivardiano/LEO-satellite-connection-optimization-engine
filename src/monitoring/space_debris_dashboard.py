#!/usr/bin/env python3
"""
Space Debris Visualization Dashboard

This module provides a comprehensive dashboard for visualizing:
- Space debris objects in LEO/MEO/GEO orbits
- Active satellites and their trajectories
- Ground station networks
- Wired network infrastructure
- Collision risk assessment
- Real-time tracking and alerts
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
import pydeck as pdk
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.leo_simulation.geometry import SatelliteGeometry
from src.monitoring.debris_tracker import DebrisTracker
from src.monitoring.network_topology import NetworkTopology

logger = logging.getLogger(__name__)

class SpaceDebrisVisualizationDashboard:
    """
    Main dashboard class for space debris visualization with integrated
    satellite tracking, ground station networks, and wired infrastructure.
    """
    
    def __init__(self):
        """Initialize the dashboard with all necessary components."""
        self.debris_tracker = DebrisTracker()
        self.satellite_geometry = SatelliteGeometry()
        self.network_topology = NetworkTopology()
        
        # Dashboard configuration
        self.update_interval = 30  # seconds
        self.max_debris_objects = 5000  # Limit for performance
        self.collision_threshold_km = 50  # km
        
        # Color schemes
        self.colors = {
            'debris': '#FF6B6B',
            'active_satellites': '#4ECDC4',
            'ground_stations': '#45B7D1',
            'wired_network': '#96CEB4',
            'collision_risk': '#FECA57',
            'critical_alert': '#FF3838'
        }
    
    def setup_dashboard(self):
        """Set up the Streamlit dashboard layout and configuration."""
        st.set_page_config(
            page_title="Space Debris Visualization Dashboard",
            page_icon="🛰️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .alert-critical {
            background: #ff4757;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .alert-warning {
            background: #ffa502;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_header(self):
        """Render the main dashboard header."""
        st.markdown("""
        <div class="main-header">
            <h1>🛰️ Space Debris Visualization Dashboard</h1>
            <p>Real-time monitoring of satellites, debris, and network infrastructure</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and filters."""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Real-time", "Last 1 hour", "Last 6 hours", "Last 24 hours", "Custom"]
        )
        
        # Object type filters
        st.sidebar.subheader("🔍 Object Filters")
        show_debris = st.sidebar.checkbox("Space Debris", value=True)
        show_active_sats = st.sidebar.checkbox("Active Satellites", value=True)
        show_ground_stations = st.sidebar.checkbox("Ground Stations", value=True)
        show_wired_network = st.sidebar.checkbox("Wired Network", value=True)
        
        # Orbit filters
        st.sidebar.subheader("🌍 Orbital Filters")
        show_leo = st.sidebar.checkbox("LEO (160-2000 km)", value=True)
        show_meo = st.sidebar.checkbox("MEO (2000-35786 km)", value=False)
        show_geo = st.sidebar.checkbox("GEO (35786 km)", value=False)
        
        # Risk assessment
        st.sidebar.subheader("⚠️ Risk Assessment")
        collision_threshold = st.sidebar.slider(
            "Collision Alert Threshold (km)", 
            min_value=1, 
            max_value=100, 
            value=self.collision_threshold_km
        )
        
        # Update settings
        st.sidebar.subheader("🔄 Update Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)", 
                min_value=10, 
                max_value=300, 
                value=self.update_interval
            )
        
        return {
            'time_range': time_range,
            'show_debris': show_debris,
            'show_active_sats': show_active_sats,
            'show_ground_stations': show_ground_stations,
            'show_wired_network': show_wired_network,
            'show_leo': show_leo,
            'show_meo': show_meo,
            'show_geo': show_geo,
            'collision_threshold': collision_threshold,
            'auto_refresh': auto_refresh
        }
    
    def render_metrics_overview(self, data: Dict):
        """Render key metrics in cards."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="🗑️ Tracked Debris",
                value=f"{data.get('debris_count', 0):,}",
                delta=f"+{data.get('debris_delta', 0)}"
            )
        
        with col2:
            st.metric(
                label="🛰️ Active Satellites",
                value=f"{data.get('active_satellites', 0):,}",
                delta=f"+{data.get('satellites_delta', 0)}"
            )
        
        with col3:
            st.metric(
                label="📡 Ground Stations",
                value=f"{data.get('ground_stations', 0):,}",
                delta=f"+{data.get('stations_delta', 0)}"
            )
        
        with col4:
            st.metric(
                label="⚠️ Collision Risks",
                value=f"{data.get('collision_risks', 0):,}",
                delta=f"{data.get('risk_delta', 0):+d}",
                delta_color="inverse"
            )
        
        with col5:
            st.metric(
                label="🌐 Network Links",
                value=f"{data.get('network_links', 0):,}",
                delta=f"+{data.get('links_delta', 0)}"
            )
    
    def create_3d_orbital_view(self, debris_data: pd.DataFrame, 
                              satellite_data: pd.DataFrame,
                              ground_stations: pd.DataFrame) -> go.Figure:
        """Create a 3D visualization of orbital objects and ground infrastructure."""
        fig = go.Figure()
        
        # Add Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = 6371 * np.outer(np.cos(u), np.sin(v))
        y_earth = 6371 * np.outer(np.sin(u), np.sin(v))
        z_earth = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            opacity=0.3,
            showscale=False,
            name='Earth'
        ))
        
        # Add space debris
        if not debris_data.empty:
            fig.add_trace(go.Scatter3d(
                x=debris_data['x_km'],
                y=debris_data['y_km'],
                z=debris_data['z_km'],
                mode='markers',
                marker=dict(
                    size=3,
                    color=self.colors['debris'],
                    opacity=0.6
                ),
                name='Space Debris',
                text=debris_data['object_name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.0f} km<br>' +
                             'Y: %{y:.0f} km<br>' +
                             'Z: %{z:.0f} km<extra></extra>'
            ))
        
        # Add active satellites
        if not satellite_data.empty:
            fig.add_trace(go.Scatter3d(
                x=satellite_data['x_km'],
                y=satellite_data['y_km'],
                z=satellite_data['z_km'],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.colors['active_satellites'],
                    opacity=0.8,
                    symbol='diamond'
                ),
                name='Active Satellites',
                text=satellite_data['satellite_name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.0f} km<br>' +
                             'Y: %{y:.0f} km<br>' +
                             'Z: %{z:.0f} km<br>' +
                             'Altitude: %{customdata[0]:.0f} km<extra></extra>',
                customdata=satellite_data[['altitude_km']].values
            ))
        
        # Add ground stations (projected to Earth surface)
        if not ground_stations.empty:
            # Convert lat/lon to 3D coordinates on Earth surface
            lat_rad = np.radians(ground_stations['latitude'])
            lon_rad = np.radians(ground_stations['longitude'])
            
            x_gs = 6371 * np.cos(lat_rad) * np.cos(lon_rad)
            y_gs = 6371 * np.cos(lat_rad) * np.sin(lon_rad)
            z_gs = 6371 * np.sin(lat_rad)
            
            fig.add_trace(go.Scatter3d(
                x=x_gs,
                y=y_gs,
                z=z_gs,
                mode='markers',
                marker=dict(
                    size=8,
                    color=self.colors['ground_stations'],
                    symbol='square',
                    opacity=0.9
                ),
                name='Ground Stations',
                text=ground_stations['station_name'],
                hovertemplate='<b>%{text}</b><br>' +
                             'Lat: %{customdata[0]:.2f}°<br>' +
                             'Lon: %{customdata[1]:.2f}°<extra></extra>',
                customdata=ground_stations[['latitude', 'longitude']].values
            ))
        
        # Update layout
        fig.update_layout(
            title="3D Orbital View - Real-time Space Environment",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                ),
                aspectmode='cube'
            ),
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_ground_network_map(self, ground_stations: pd.DataFrame,
                                 wired_connections: pd.DataFrame) -> folium.Map:
        """Create a folium map showing ground station network and wired connections."""
        # Center map on world view
        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles='CartoDB dark_matter'
        )
        
        # Add ground stations
        for _, station in ground_stations.iterrows():
            folium.CircleMarker(
                location=[station['latitude'], station['longitude']],
                radius=8,
                popup=f"<b>{station['station_name']}</b><br>"
                      f"Lat: {station['latitude']:.2f}°<br>"
                      f"Lon: {station['longitude']:.2f}°<br>"
                      f"Type: {station.get('station_type', 'Unknown')}",
                color='#45B7D1',
                fill=True,
                fillColor='#45B7D1',
                fillOpacity=0.8
            ).add_to(m)
        
        # Add wired network connections
        for _, connection in wired_connections.iterrows():
            folium.PolyLine(
                locations=[
                    [connection['start_lat'], connection['start_lon']],
                    [connection['end_lat'], connection['end_lon']]
                ],
                color='#96CEB4',
                weight=3,
                opacity=0.7,
                popup=f"Link: {connection['connection_type']}<br>"
                      f"Capacity: {connection.get('capacity_gbps', 'Unknown')} Gbps"
            ).add_to(m)
        
        return m
    
    def create_collision_risk_chart(self, risk_data: pd.DataFrame) -> go.Figure:
        """Create a chart showing collision risk assessment over time."""
        fig = go.Figure()
        
        # Add risk levels
        risk_levels = ['Low', 'Medium', 'High', 'Critical']
        colors = ['#2ECC71', '#F39C12', '#E74C3C', '#8E44AD']
        
        for i, level in enumerate(risk_levels):
            level_data = risk_data[risk_data['risk_level'] == level]
            if not level_data.empty:
                fig.add_trace(go.Scatter(
                    x=level_data['timestamp'],
                    y=level_data['object_count'],
                    mode='lines+markers',
                    name=f'{level} Risk',
                    line=dict(color=colors[i], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title="Collision Risk Assessment Timeline",
            xaxis_title="Time",
            yaxis_title="Number of Objects at Risk",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def render_alerts_panel(self, alerts: List[Dict]):
        """Render the alerts and notifications panel."""
        st.subheader("🚨 Active Alerts")
        
        if not alerts:
            st.success("✅ No active alerts - All systems nominal")
            return
        
        # Sort alerts by severity
        alerts_sorted = sorted(alerts, key=lambda x: x.get('severity', 0), reverse=True)
        
        for alert in alerts_sorted[:10]:  # Show top 10 alerts
            severity = alert.get('severity', 1)
            alert_type = alert.get('type', 'Unknown')
            message = alert.get('message', 'No details available')
            timestamp = alert.get('timestamp', datetime.now())
            
            if severity >= 4:  # Critical
                st.markdown(f"""
                <div class="alert-critical">
                    <strong>🚨 CRITICAL: {alert_type}</strong><br>
                    {message}<br>
                    <small>Time: {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
            elif severity >= 2:  # Warning
                st.markdown(f"""
                <div class="alert-warning">
                    <strong>⚠️ WARNING: {alert_type}</strong><br>
                    {message}<br>
                    <small>Time: {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)
            else:  # Info
                st.info(f"ℹ️ **{alert_type}**: {message}")
    
    def run_dashboard(self):
        """Main method to run the dashboard."""
        try:
            # Setup dashboard
            self.setup_dashboard()
            self.render_header()
            
            # Get sidebar controls
            controls = self.render_sidebar()
            
            # Load data based on controls
            with st.spinner("Loading space environment data..."):
                data = self.load_dashboard_data(controls)
            
            # Render metrics overview
            self.render_metrics_overview(data['metrics'])
            
            # Create main dashboard tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "🌍 3D Orbital View",
                "🗺️ Ground Network",
                "📊 Risk Analysis",
                "🚨 Alerts",
                "⚙️ System Status"
            ])
            
            with tab1:
                st.subheader("3D Orbital Environment")
                fig_3d = self.create_3d_orbital_view(
                    data['debris_data'],
                    data['satellite_data'],
                    data['ground_stations']
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Add orbital statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Debris Distribution by Altitude**")
                    debris_alt_dist = data['debris_data'].groupby(
                        pd.cut(data['debris_data']['altitude_km'], 
                               bins=[0, 500, 1000, 2000, 35786, 50000])
                    ).size()
                    st.bar_chart(debris_alt_dist)
                
                with col2:
                    st.write("**Satellite Types**")
                    sat_types = data['satellite_data']['satellite_type'].value_counts()
                    st.pie_chart(sat_types)
            
            with tab2:
                st.subheader("Ground Station Network & Wired Infrastructure")
                
                # Create and display map
                network_map = self.create_ground_network_map(
                    data['ground_stations'],
                    data['wired_connections']
                )
                st_folium(network_map, width=700, height=500)
                
                # Network statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Ground Station Status**")
                    station_status = data['ground_stations']['status'].value_counts()
                    st.bar_chart(station_status)
                
                with col2:
                    st.write("**Network Capacity Utilization**")
                    # Create capacity utilization chart
                    capacity_data = data['wired_connections'].groupby('connection_type')[
                        'utilization_percent'
                    ].mean()
                    st.bar_chart(capacity_data)
            
            with tab3:
                st.subheader("Collision Risk Analysis")
                
                # Risk timeline chart
                risk_chart = self.create_collision_risk_chart(data['risk_timeline'])
                st.plotly_chart(risk_chart, use_container_width=True)
                
                # Risk matrix
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**High-Risk Object Pairs**")
                    if not data['high_risk_pairs'].empty:
                        st.dataframe(
                            data['high_risk_pairs'][
                                ['object1', 'object2', 'distance_km', 'collision_probability']
                            ].head(10)
                        )
                    else:
                        st.info("No high-risk pairs detected")
                
                with col2:
                    st.write("**Risk Distribution**")
                    risk_dist = data['risk_timeline'].groupby('risk_level').size()
                    st.pie_chart(risk_dist)
            
            with tab4:
                self.render_alerts_panel(data['alerts'])
            
            with tab5:
                st.subheader("System Health & Performance")
                
                # System metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Update Rate", "99.8%", "+0.2%")
                    st.metric("Tracking Accuracy", "±50m", "±5m")
                
                with col2:
                    st.metric("API Response Time", "45ms", "-5ms")
                    st.metric("Database Load", "67%", "+3%")
                
                with col3:
                    st.metric("Alert Processing", "Real-time", "0ms delay")
                    st.metric("System Uptime", "99.99%", "0%")
                
                # Performance charts
                st.write("**System Performance Timeline**")
                # Add performance metrics visualization here
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {e}", exc_info=True)
    
    def load_dashboard_data(self, controls: Dict) -> Dict:
        """Load all data needed for the dashboard based on control settings."""
        try:
            # This is a placeholder - in real implementation, this would
            # fetch data from your tracking systems, databases, etc.
            
            # Generate sample data for demonstration
            data = {
                'metrics': {
                    'debris_count': 15420,
                    'debris_delta': 12,
                    'active_satellites': 3245,
                    'satellites_delta': 3,
                    'ground_stations': 156,
                    'stations_delta': 0,
                    'collision_risks': 23,
                    'risk_delta': -2,
                    'network_links': 89,
                    'links_delta': 1
                },
                'debris_data': self._generate_sample_debris_data(),
                'satellite_data': self._generate_sample_satellite_data(),
                'ground_stations': self._generate_sample_ground_stations(),
                'wired_connections': self._generate_sample_wired_connections(),
                'risk_timeline': self._generate_sample_risk_data(),
                'high_risk_pairs': self._generate_sample_risk_pairs(),
                'alerts': self._generate_sample_alerts()
            }
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading dashboard data: {e}")
            raise
    
    def _generate_sample_debris_data(self) -> pd.DataFrame:
        """Generate sample debris data for demonstration."""
        n_debris = 1000
        
        # Generate orbital positions
        np.random.seed(42)
        altitudes = np.random.exponential(scale=800, size=n_debris) + 200  # LEO focus
        inclinations = np.random.uniform(0, 180, n_debris)
        raans = np.random.uniform(0, 360, n_debris)
        
        # Convert to Cartesian coordinates (simplified)
        r = 6371 + altitudes  # Earth radius + altitude
        x = r * np.cos(np.radians(raans)) * np.cos(np.radians(inclinations))
        y = r * np.sin(np.radians(raans)) * np.cos(np.radians(inclinations))
        z = r * np.sin(np.radians(inclinations))
        
        return pd.DataFrame({
            'object_name': [f'DEBRIS-{i:04d}' for i in range(n_debris)],
            'x_km': x,
            'y_km': y,
            'z_km': z,
            'altitude_km': altitudes,
            'object_type': np.random.choice(['Fragment', 'Rocket Body', 'Defunct Satellite'], n_debris)
        })
    
    def _generate_sample_satellite_data(self) -> pd.DataFrame:
        """Generate sample active satellite data."""
        satellites = [
            {'name': 'ISS (ZARYA)', 'alt': 408, 'type': 'Space Station'},
            {'name': 'STARLINK-1234', 'alt': 550, 'type': 'Communication'},
            {'name': 'TERRA', 'alt': 705, 'type': 'Earth Observation'},
            {'name': 'NOAA-18', 'alt': 854, 'type': 'Weather'},
            {'name': 'GPS BIIR-2', 'alt': 20200, 'type': 'Navigation'},
        ]
        
        data = []
        for sat in satellites:
            # Generate position (simplified orbital mechanics)
            alt = sat['alt']
            r = 6371 + alt
            angle = np.random.uniform(0, 2*np.pi)
            inclination = np.random.uniform(0, np.pi)
            
            x = r * np.cos(angle) * np.cos(inclination)
            y = r * np.sin(angle) * np.cos(inclination)
            z = r * np.sin(inclination)
            
            data.append({
                'satellite_name': sat['name'],
                'x_km': x,
                'y_km': y,
                'z_km': z,
                'altitude_km': alt,
                'satellite_type': sat['type']
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_ground_stations(self) -> pd.DataFrame:
        """Generate sample ground station data."""
        stations = [
            {'name': 'Goldstone', 'lat': 35.4, 'lon': -116.9, 'type': 'Deep Space'},
            {'name': 'Canberra', 'lat': -35.4, 'lon': 148.9, 'type': 'Deep Space'},
            {'name': 'Madrid', 'lat': 40.4, 'lon': -4.2, 'type': 'Deep Space'},
            {'name': 'Kourou', 'lat': 5.2, 'lon': -52.8, 'type': 'Launch'},
            {'name': 'Svalbard', 'lat': 78.2, 'lon': 15.4, 'type': 'Polar'},
            {'name': 'Hawaii', 'lat': 21.3, 'lon': -157.8, 'type': 'Tracking'},
        ]
        
        return pd.DataFrame([
            {
                'station_name': s['name'],
                'latitude': s['lat'],
                'longitude': s['lon'],
                'station_type': s['type'],
                'status': np.random.choice(['Online', 'Maintenance', 'Offline'], p=[0.8, 0.15, 0.05])
            }
            for s in stations
        ])
    
    def _generate_sample_wired_connections(self) -> pd.DataFrame:
        """Generate sample wired network connections."""
        connections = [
            {'start': 'Goldstone', 'end': 'Canberra', 'type': 'Fiber Optic'},
            {'start': 'Canberra', 'end': 'Madrid', 'type': 'Fiber Optic'},
            {'start': 'Madrid', 'end': 'Kourou', 'type': 'Satellite Link'},
            {'start': 'Kourou', 'end': 'Hawaii', 'type': 'Undersea Cable'},
            {'start': 'Hawaii', 'end': 'Svalbard', 'type': 'Satellite Link'},
        ]
        
        # Map to coordinates (simplified)
        coords = {
            'Goldstone': (35.4, -116.9),
            'Canberra': (-35.4, 148.9),
            'Madrid': (40.4, -4.2),
            'Kourou': (5.2, -52.8),
            'Hawaii': (21.3, -157.8),
            'Svalbard': (78.2, 15.4)
        }
        
        data = []
        for conn in connections:
            start_coords = coords[conn['start']]
            end_coords = coords[conn['end']]
            
            data.append({
                'connection_type': conn['type'],
                'start_lat': start_coords[0],
                'start_lon': start_coords[1],
                'end_lat': end_coords[0],
                'end_lon': end_coords[1],
                'capacity_gbps': np.random.randint(10, 100),
                'utilization_percent': np.random.randint(30, 90)
            })
        
        return pd.DataFrame(data)
    
    def _generate_sample_risk_data(self) -> pd.DataFrame:
        """Generate sample collision risk timeline data."""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='H'
        )
        
        data = []
        for ts in timestamps:
            for level in ['Low', 'Medium', 'High', 'Critical']:
                count = np.random.poisson(
                    {'Low': 50, 'Medium': 15, 'High': 5, 'Critical': 1}[level]
                )
                data.append({
                    'timestamp': ts,
                    'risk_level': level,
                    'object_count': count
                })
        
        return pd.DataFrame(data)
    
    def _generate_sample_risk_pairs(self) -> pd.DataFrame:
        """Generate sample high-risk object pairs."""
        pairs = []
        for i in range(10):
            pairs.append({
                'object1': f'DEBRIS-{np.random.randint(1000, 9999)}',
                'object2': f'SAT-{np.random.randint(100, 999)}',
                'distance_km': np.random.uniform(1, 50),
                'collision_probability': np.random.uniform(0.001, 0.1),
                'time_to_closest_approach': f'{np.random.randint(1, 72)}h'
            })
        
        return pd.DataFrame(pairs)
    
    def _generate_sample_alerts(self) -> List[Dict]:
        """Generate sample alerts."""
        return [
            {
                'type': 'Collision Warning',
                'message': 'STARLINK-1234 approaching DEBRIS-5678 - 15km separation in 2 hours',
                'severity': 3,
                'timestamp': datetime.now() - timedelta(minutes=5)
            },
            {
                'type': 'Ground Station Offline',
                'message': 'Madrid ground station communication lost',
                'severity': 2,
                'timestamp': datetime.now() - timedelta(minutes=15)
            },
            {
                'type': 'New Debris Detected',
                'message': '5 new debris objects detected in LEO orbit',
                'severity': 1,
                'timestamp': datetime.now() - timedelta(minutes=30)
            }
        ]


def main():
    """Main function to run the dashboard."""
    dashboard = SpaceDebrisVisualizationDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
