#!/usr/bin/env python3
"""
Simple Space Debris Visualization Dashboard

A streamlined version of the space debris dashboard that works without
complex dependencies.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import folium
from streamlit_folium import st_folium
from typing import Dict, List, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)

class SimpleDebrisVisualizationDashboard:
    """
    Simplified space debris visualization dashboard.
    """
    
    def __init__(self):
        """Initialize the dashboard with sample data."""
        # Dashboard configuration
        self.update_interval = 30  # seconds
        self.max_debris_objects = 1000  # Limit for performance
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
            ["Real-time", "Last 1 hour", "Last 6 hours", "Last 24 hours"]
        )
        
        # Object type filters
        st.sidebar.subheader("🔍 Object Filters")
        show_debris = st.sidebar.checkbox("Space Debris", value=True)
        show_active_sats = st.sidebar.checkbox("Active Satellites", value=True)
        show_ground_stations = st.sidebar.checkbox("Ground Stations", value=True)
        
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
        
        return {
            'time_range': time_range,
            'show_debris': show_debris,
            'show_active_sats': show_active_sats,
            'show_ground_stations': show_ground_stations,
            'show_leo': show_leo,
            'show_meo': show_meo,
            'show_geo': show_geo,
            'collision_threshold': collision_threshold
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
        """Create a 3D visualization of orbital objects."""
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
    
    def create_ground_network_map(self, ground_stations: pd.DataFrame) -> folium.Map:
        """Create a folium map showing ground station network."""
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
    
    def generate_sample_debris_data(self) -> pd.DataFrame:
        """Generate sample debris data for demonstration."""
        n_debris = 500
        
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
    
    def generate_sample_satellite_data(self) -> pd.DataFrame:
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
    
    def generate_sample_ground_stations(self) -> pd.DataFrame:
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
    
    def generate_sample_risk_data(self) -> pd.DataFrame:
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
    
    def run_dashboard(self):
        """Main method to run the dashboard."""
        try:
            # Setup dashboard
            self.setup_dashboard()
            self.render_header()
            
            # Get sidebar controls
            controls = self.render_sidebar()
            
            # Generate sample data
            with st.spinner("Loading space environment data..."):
                debris_data = self.generate_sample_debris_data()
                satellite_data = self.generate_sample_satellite_data()
                ground_stations = self.generate_sample_ground_stations()
                risk_data = self.generate_sample_risk_data()
            
            # Sample metrics
            metrics = {
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
            }
            
            # Render metrics overview
            self.render_metrics_overview(metrics)
            
            # Create main dashboard tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌍 3D Orbital View",
                "🗺️ Ground Network",
                "📊 Risk Analysis",
                "🚨 System Status"
            ])
            
            with tab1:
                st.subheader("3D Orbital Environment")
                fig_3d = self.create_3d_orbital_view(
                    debris_data,
                    satellite_data,
                    ground_stations
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Add orbital statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Debris Distribution by Altitude**")
                    debris_alt_dist = debris_data.groupby(
                        pd.cut(debris_data['altitude_km'], 
                               bins=[0, 500, 1000, 2000, 35786, 50000])
                    ).size()
                    st.bar_chart(debris_alt_dist)
                
                with col2:
                    st.write("**Satellite Types**")
                    sat_types = satellite_data['satellite_type'].value_counts()
                    fig_pie = px.pie(values=sat_types.values, names=sat_types.index)
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with tab2:
                st.subheader("Ground Station Network")
                
                # Create and display map
                network_map = self.create_ground_network_map(ground_stations)
                st_folium(network_map, width=700, height=500)
                
                # Network statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Ground Station Status**")
                    station_status = ground_stations['status'].value_counts()
                    st.bar_chart(station_status)
                
                with col2:
                    st.write("**Station Types**")
                    station_types = ground_stations['station_type'].value_counts()
                    st.bar_chart(station_types)
            
            with tab3:
                st.subheader("Collision Risk Analysis")
                
                # Risk timeline chart
                risk_chart = self.create_collision_risk_chart(risk_data)
                st.plotly_chart(risk_chart, use_container_width=True)
                
                # Risk statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Current Risk Distribution**")
                    current_risks = risk_data[risk_data['timestamp'] == risk_data['timestamp'].max()]
                    risk_dist = current_risks.groupby('risk_level')['object_count'].sum()
                    fig_risk_pie = px.pie(values=risk_dist.values, names=risk_dist.index)
                    st.plotly_chart(fig_risk_pie, use_container_width=True)
                
                with col2:
                    st.write("**Risk Summary**")
                    st.info("✅ No critical collision risks detected")
                    st.warning("⚠️ 5 medium-risk object pairs monitored")
                    st.success("📊 Risk assessment updated every 30 seconds")
            
            with tab4:
                st.subheader("System Health & Performance")
                
                # System metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Update Rate", "99.8%", "+0.2%")
                    st.metric("Tracking Accuracy", "±50m", "±5m")
                
                with col2:
                    st.metric("Dashboard Response", "45ms", "-5ms")
                    st.metric("Active Connections", "67", "+3")
                
                with col3:
                    st.metric("Alert Processing", "Real-time", "0ms delay")
                    st.metric("System Uptime", "99.99%", "0%")
                
                # Status messages
                st.success("🟢 All systems operational")
                st.info("🔄 Next TLE update in 2 hours")
                st.info("📡 Tracking 15,420 objects across all orbits")
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {e}", exc_info=True)


def main():
    """Main function to run the dashboard."""
    dashboard = SimpleDebrisVisualizationDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
