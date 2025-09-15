#!/usr/bin/env python3
"""
Minimal Space Debris Visualization Dashboard

A basic version that works with minimal dependencies.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MinimalDebrisVisualizationDashboard:
    """
    Minimal space debris visualization dashboard.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.colors = {
            'debris': '#FF6B6B',
            'active_satellites': '#4ECDC4',
            'ground_stations': '#45B7D1',
            'collision_risk': '#FECA57',
            'critical_alert': '#FF3838'
        }
    
    def setup_dashboard(self):
        """Set up the Streamlit dashboard."""
        st.set_page_config(
            page_title="🛰️ Space Debris Dashboard",
            page_icon="🛰️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
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
        """Render the sidebar with controls."""
        st.sidebar.header("🎛️ Dashboard Controls")
        
        time_range = st.sidebar.selectbox(
            "Time Range",
            ["Real-time", "Last 1 hour", "Last 6 hours", "Last 24 hours"]
        )
        
        st.sidebar.subheader("🔍 Object Filters")
        show_debris = st.sidebar.checkbox("Space Debris", value=True)
        show_satellites = st.sidebar.checkbox("Active Satellites", value=True)
        show_ground_stations = st.sidebar.checkbox("Ground Stations", value=True)
        
        st.sidebar.subheader("🌍 Orbital Filters")
        show_leo = st.sidebar.checkbox("LEO (160-2000 km)", value=True)
        show_meo = st.sidebar.checkbox("MEO (2000-35786 km)", value=False)
        show_geo = st.sidebar.checkbox("GEO (35786 km)", value=False)
        
        collision_threshold = st.sidebar.slider(
            "Collision Alert Threshold (km)", 
            min_value=1, 
            max_value=100, 
            value=50
        )
        
        return {
            'time_range': time_range,
            'show_debris': show_debris,
            'show_satellites': show_satellites,
            'show_ground_stations': show_ground_stations,
            'show_leo': show_leo,
            'show_meo': show_meo,
            'show_geo': show_geo,
            'collision_threshold': collision_threshold
        }
    
    def render_metrics(self, controls: Dict):
        """Render key metrics based on current filters."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        # Calculate filtered counts
        debris_count = 15420 if controls['show_debris'] else 0
        satellite_count = 3245 if controls['show_satellites'] else 0  
        station_count = 156 if controls['show_ground_stations'] else 0
        
        # Simulate collision risks based on threshold
        base_risks = 50
        risk_multiplier = max(0.1, (100 - controls['collision_threshold']) / 100)
        collision_risks = int(base_risks * risk_multiplier)
        
        with col1:
            st.metric("🗑️ Tracked Debris", f"{debris_count:,}", "+12" if controls['show_debris'] else "Hidden")
        with col2:
            st.metric("🛰️ Active Satellites", f"{satellite_count:,}", "+3" if controls['show_satellites'] else "Hidden")
        with col3:
            st.metric("📡 Ground Stations", f"{station_count:,}", "0" if controls['show_ground_stations'] else "Hidden")
        with col4:
            st.metric("⚠️ Collision Risks", f"{collision_risks}", f"Threshold: {controls['collision_threshold']}km")
        with col5:
            network_links = 89 if any([controls['show_debris'], controls['show_satellites'], controls['show_ground_stations']]) else 0
            st.metric("🌐 Network Links", f"{network_links}", "+1" if network_links > 0 else "No data")
    
    def create_3d_orbital_view(self, controls: Dict) -> go.Figure:
        """Create a 3D visualization of orbital objects."""
        fig = go.Figure()
        
        # Add realistic Earth sphere
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)
        x_earth = 6371 * np.outer(np.cos(u), np.sin(v))
        y_earth = 6371 * np.outer(np.sin(u), np.sin(v))
        z_earth = 6371 * np.outer(np.ones(np.size(u)), np.cos(v))
        
        # Create realistic Earth coloring based on lat/lon
        # This simulates continents (green/brown) and oceans (blue)
        earth_colors = np.zeros((len(v), len(u)))
        
        for i, theta in enumerate(v):  # latitude
            for j, phi in enumerate(u):  # longitude
                lat = 90 - np.degrees(theta)  # Convert to latitude (-90 to 90)
                lon = np.degrees(phi) - 180   # Convert to longitude (-180 to 180)
                
                # Simulate major continents and oceans
                # This is a simplified representation
                is_land = False
                
                # North America
                if -170 < lon < -50 and 25 < lat < 75:
                    is_land = True
                # South America  
                elif -85 < lon < -35 and -55 < lat < 15:
                    is_land = True
                # Europe
                elif -10 < lon < 40 and 35 < lat < 70:
                    is_land = True
                # Africa
                elif -20 < lon < 50 and -35 < lat < 37:
                    is_land = True
                # Asia
                elif 25 < lon < 180 and 10 < lat < 80:
                    is_land = True
                # Australia
                elif 110 < lon < 160 and -45 < lat < -10:
                    is_land = True
                # Antarctica (rough approximation)
                elif lat < -60:
                    is_land = True
                
                if is_land:
                    # Land colors (green to brown gradient)
                    if lat > 60:  # Arctic
                        earth_colors[i, j] = 0.9  # White/light
                    elif lat > 40:  # Temperate
                        earth_colors[i, j] = 0.6  # Green
                    elif lat > -40:  # Tropical/temperate
                        earth_colors[i, j] = 0.7  # Light green/brown
                    else:  # Antarctic
                        earth_colors[i, j] = 0.95  # White
                else:
                    # Ocean colors (deep blue)
                    earth_colors[i, j] = 0.1  # Dark blue
        
        fig.add_trace(go.Surface(
            x=x_earth, 
            y=y_earth, 
            z=z_earth,
            surfacecolor=earth_colors,
            colorscale=[
                [0.0, '#003366'],    # Deep ocean blue
                [0.1, '#0066CC'],    # Ocean blue
                [0.2, '#4D9900'],    # Forest green
                [0.6, '#66B266'],    # Light green
                [0.7, '#8B4513'],    # Brown (land)
                [0.9, '#F5F5DC'],    # Beige (desert/arctic)
                [1.0, '#FFFFFF']     # White (ice/snow)
            ],
            opacity=0.9,
            showscale=False,
            name='Earth',
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                fresnel=0.1,
                specular=0.05,
                roughness=0.1
            ),
            lightposition=dict(x=100, y=200, z=300)
        ))
        
        # Generate and filter debris data based on controls
        if controls['show_debris']:
            n_debris = 200
            np.random.seed(42)
            altitudes = np.random.exponential(scale=800, size=n_debris) + 200
            
            # Filter by orbital regions
            filtered_debris = []
            for i, alt in enumerate(altitudes):
                show_this = False
                if controls['show_leo'] and 160 <= alt <= 2000:
                    show_this = True
                elif controls['show_meo'] and 2000 < alt <= 35786:
                    show_this = True
                elif controls['show_geo'] and alt > 35786:
                    show_this = True
                
                if show_this:
                    filtered_debris.append(i)
            
            if filtered_debris:
                filtered_altitudes = altitudes[filtered_debris]
                angles = np.random.uniform(0, 2*np.pi, len(filtered_debris))
                inclinations = np.random.uniform(0, np.pi, len(filtered_debris))
                
                r = 6371 + filtered_altitudes
                x_debris = r * np.cos(angles) * np.sin(inclinations)
                y_debris = r * np.sin(angles) * np.sin(inclinations)
                z_debris = r * np.cos(inclinations)
                
                # Color code by collision risk based on threshold
                colors_debris = []
                for alt in filtered_altitudes:
                    # Simulate collision risk based on altitude and threshold
                    risk_distance = np.random.uniform(1, 100)
                    if risk_distance < controls['collision_threshold']:
                        colors_debris.append(self.colors['collision_risk'])  # Yellow for risk
                    else:
                        colors_debris.append(self.colors['debris'])  # Red for normal
                
                fig.add_trace(go.Scatter3d(
                    x=x_debris,
                    y=y_debris,
                    z=z_debris,
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=colors_debris,
                        opacity=0.6
                    ),
                    name=f'Space Debris ({len(filtered_debris)})',
                    text=[f'DEBRIS-{filtered_debris[i]:04d}' for i in range(len(filtered_debris))],
                    hovertemplate='<b>%{text}</b><br>Altitude: %{customdata:.0f} km<extra></extra>',
                    customdata=filtered_altitudes
                ))
        
        # Add satellites with realistic orbits and movement
        if controls['show_satellites']:
            # Get current time for realistic orbital calculations with offset
            import time
            current_time = time.time()
            
            # Add time offset from session state for position updates
            if 'time_offset' in st.session_state:
                current_time += st.session_state.time_offset
            
            # Use actual time for realistic orbital motion - amplified for visibility
            time_factor = current_time / 3600  # Hours since epoch
            
            # Add visual amplification factor to make movement more visible
            movement_amplification = 1
            if 'time_offset' in st.session_state and st.session_state.time_offset > 0:
                movement_amplification = 5  # Make movement 5x more visible when time offset is active
            
            satellites = [
                {'name': 'ISS', 'alt': 408, 'inclination': 51.6, 'period': 93, 'color': '#FF6B35', 'type': 'Space Station'},
                {'name': 'STARLINK-1234', 'alt': 550, 'inclination': 53.0, 'period': 96, 'color': '#4ECDC4', 'type': 'Communication'},
                {'name': 'TERRA', 'alt': 705, 'inclination': 98.2, 'period': 99, 'color': '#45B7D1', 'type': 'Earth Observation'},
                {'name': 'NOAA-18', 'alt': 854, 'inclination': 99.2, 'period': 102, 'color': '#96CEB4', 'type': 'Weather'},
                {'name': 'GPS-BIIR', 'alt': 20200, 'inclination': 55.0, 'period': 718, 'color': '#FECA57', 'type': 'Navigation'},
                {'name': 'GOES-16', 'alt': 35786, 'inclination': 0.1, 'period': 1436, 'color': '#FF9FF3', 'type': 'Geostationary'},
            ]
            
            for i, sat in enumerate(satellites):
                # Filter by orbital regions
                show_sat = False
                alt = sat['alt']
                if controls['show_leo'] and 160 <= alt <= 2000:
                    show_sat = True
                elif controls['show_meo'] and 2000 < alt <= 35786:
                    show_sat = True
                elif controls['show_geo'] and alt > 35786:
                    show_sat = True
                
                if show_sat:
                    # Calculate realistic orbital position
                    r_sat = 6371 + alt
                    
                    # Orbital parameters
                    inclination_rad = np.radians(sat['inclination'])
                    
                    # Calculate realistic orbital motion based on actual orbital periods
                    # Each satellite moves at its real orbital speed
                    orbital_period_hours = sat['period'] / 60  # Convert minutes to hours
                    
                    # Current orbital position (realistic speed with amplification)
                    # Satellites complete full orbit in their actual period
                    mean_anomaly = (time_factor * movement_amplification * 2 * np.pi / orbital_period_hours + i * np.pi/3) % (2 * np.pi)
                    
                    # Calculate position in orbital plane
                    x_orbit = r_sat * np.cos(mean_anomaly)
                    y_orbit = r_sat * np.sin(mean_anomaly)
                    z_orbit = 0
                    
                    # Apply orbital inclination (rotate around x-axis)
                    x_sat = x_orbit
                    y_sat = y_orbit * np.cos(inclination_rad) - z_orbit * np.sin(inclination_rad)
                    z_sat = y_orbit * np.sin(inclination_rad) + z_orbit * np.cos(inclination_rad)
                    
                    # Apply rotation around Earth's axis (longitude)
                    earth_rotation = time_factor * 2 * np.pi * 0.1  # Slow Earth rotation effect
                    
                    x_final = x_sat * np.cos(earth_rotation) - y_sat * np.sin(earth_rotation)
                    y_final = x_sat * np.sin(earth_rotation) + y_sat * np.cos(earth_rotation)
                    z_final = z_sat
                    
                    # Orbital velocity calculation (for future use)
                    orbital_velocity = 2 * np.pi * r_sat / (orbital_period_hours * 3600)  # km/s
                    
                    # Add satellite marker
                    fig.add_trace(go.Scatter3d(
                        x=[x_final],
                        y=[y_final],
                        z=[z_final],
                        mode='markers',
                        marker=dict(
                            size=10 if sat['name'] == 'ISS' else 8,
                            color=sat['color'],
                            symbol='diamond' if 'GPS' in sat['name'] or 'GOES' in sat['name'] else 'circle',
                            line=dict(width=2, color='white')
                        ),
                        name=f"{sat['name']} ({sat['type']})",
                        text=sat['name'],
                        hovertemplate=f'<b>{sat["name"]}</b><br>' +
                                     f'Type: {sat["type"]}<br>' +
                                     f'Altitude: {alt:,} km<br>' +
                                     f'Inclination: {sat["inclination"]}°<br>' +
                                     f'Period: {sat["period"]} min<extra></extra>'
                    ))
                    
                    # Add orbital trail for ISS
                    if sat['name'] == 'ISS':
                        trail_points = 10  # Reduced for performance
                        trail_angles = np.linspace(mean_anomaly - np.pi/2, mean_anomaly, trail_points)
                        
                        # Vectorized trail calculation for speed
                        x_trail = r_sat * np.cos(trail_angles)
                        y_trail = r_sat * np.sin(trail_angles)
                        z_trail = np.zeros_like(trail_angles)
                        
                        # Apply inclination (vectorized)
                        y_trail_rot = y_trail * np.cos(inclination_rad)
                        z_trail_rot = y_trail * np.sin(inclination_rad)
                        
                        # Apply Earth rotation (vectorized)
                        x_trail_final = x_trail * np.cos(earth_rotation) - y_trail_rot * np.sin(earth_rotation)
                        y_trail_final = x_trail * np.sin(earth_rotation) + y_trail_rot * np.cos(earth_rotation)
                        
                        # Add trail
                        fig.add_trace(go.Scatter3d(
                            x=x_trail_final,
                            y=y_trail_final,
                            z=z_trail_rot,
                            mode='lines',
                            line=dict(
                                color=sat['color'],
                                width=2,
                                dash='dot'
                            ),
                            opacity=0.4,
                            name=f"{sat['name']} Trail",
                            showlegend=False,
                            hoverinfo='skip'
                        ))
        
        # Add ground stations if enabled
        if controls['show_ground_stations']:
            stations = [
                {'name': 'Goldstone', 'lat': 35.4, 'lon': -116.9},
                {'name': 'Canberra', 'lat': -35.4, 'lon': 148.9},
                {'name': 'Madrid', 'lat': 40.4, 'lon': -4.2},
                {'name': 'Kourou', 'lat': 5.2, 'lon': -52.8},
                {'name': 'Svalbard', 'lat': 78.2, 'lon': 15.4},
                {'name': 'Hawaii', 'lat': 21.3, 'lon': -157.8},
            ]
            
            for station in stations:
                # Convert lat/lon to 3D coordinates on Earth surface
                lat_rad = np.radians(station['lat'])
                lon_rad = np.radians(station['lon'])
                
                x_gs = 6371 * np.cos(lat_rad) * np.cos(lon_rad)
                y_gs = 6371 * np.cos(lat_rad) * np.sin(lon_rad)
                z_gs = 6371 * np.sin(lat_rad)
                
                fig.add_trace(go.Scatter3d(
                    x=[x_gs],
                    y=[y_gs],
                    z=[z_gs],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=self.colors['ground_stations'],
                        symbol='square'
                    ),
                    name=station['name'],
                    text=station['name'],
                    hovertemplate=f'<b>{station["name"]}</b><br>Ground Station<extra></extra>'
                ))
        
        # Update title with time offset info for movement indication
        if 'time_offset' in st.session_state and st.session_state.time_offset > 0:
            hours_offset = st.session_state.time_offset / 3600
            title_time = f"Time +{hours_offset:.1f}h"
        else:
            title_time = "Real Time"
        
        fig.update_layout(
            title=f"🌍 LEO Satellite Network - 3D Orbital View ({title_time})",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)", 
                zaxis_title="Z (km)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='cube',
                bgcolor='black'  # Space background
            ),
            height=600,
            showlegend=True,
            paper_bgcolor='black',
            plot_bgcolor='black'
        )
        
        return fig
    
    def create_ground_stations_chart(self) -> go.Figure:
        """Create a ground stations visualization."""
        stations = [
            {'name': 'Goldstone', 'lat': 35.4, 'lon': -116.9, 'status': 'Online'},
            {'name': 'Canberra', 'lat': -35.4, 'lon': 148.9, 'status': 'Online'},
            {'name': 'Madrid', 'lat': 40.4, 'lon': -4.2, 'status': 'Online'},
            {'name': 'Kourou', 'lat': 5.2, 'lon': -52.8, 'status': 'Maintenance'},
            {'name': 'Svalbard', 'lat': 78.2, 'lon': 15.4, 'status': 'Online'},
            {'name': 'Hawaii', 'lat': 21.3, 'lon': -157.8, 'status': 'Online'},
        ]
        
        df = pd.DataFrame(stations)
        
        fig = go.Figure(data=go.Scattergeo(
            lon=df['lon'],
            lat=df['lat'],
            text=df['name'],
            mode='markers+text',
            marker=dict(
                size=12,
                color=[self.colors['ground_stations'] if s == 'Online' else '#FFA500' for s in df['status']],
                line=dict(width=2, color='white')
            ),
            textposition="top center"
        ))
        
        fig.update_layout(
            title="Global Ground Station Network",
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='rgb(243, 243, 243)',
                coastlinecolor='rgb(204, 204, 204)',
            ),
            height=500
        )
        
        return fig
    
    def create_collision_risk_chart(self) -> go.Figure:
        """Create collision risk timeline."""
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=24),
            end=datetime.now(),
            freq='h'
        )
        
        risk_data = []
        for ts in timestamps:
            for level in ['Low', 'Medium', 'High', 'Critical']:
                count = np.random.poisson(
                    {'Low': 50, 'Medium': 15, 'High': 5, 'Critical': 1}[level]
                )
                risk_data.append({
                    'timestamp': ts,
                    'risk_level': level,
                    'count': count
                })
        
        df = pd.DataFrame(risk_data)
        
        fig = go.Figure()
        
        colors = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E74C3C', 'Critical': '#8E44AD'}
        
        for level in ['Low', 'Medium', 'High', 'Critical']:
            level_data = df[df['risk_level'] == level]
            fig.add_trace(go.Scatter(
                x=level_data['timestamp'],
                y=level_data['count'],
                mode='lines+markers',
                name=f'{level} Risk',
                line=dict(color=colors[level], width=2)
            ))
        
        fig.update_layout(
            title="Collision Risk Assessment Timeline",
            xaxis_title="Time",
            yaxis_title="Objects at Risk",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def run_dashboard(self):
        """Main dashboard function."""
        try:
            self.setup_dashboard()
            self.render_header()
            
            controls = self.render_sidebar()
            
            # Add satellite movement control with working position updates
            movement_enabled = st.sidebar.checkbox("🛰️ Enable Satellite Movement", value=False)
            
            # Initialize time offset in session state
            if 'time_offset' not in st.session_state:
                st.session_state.time_offset = 0
            
            if movement_enabled:
                # Add time step selector for different movement speeds
                time_step = st.sidebar.selectbox(
                    "⏱️ Time Step per Click:",
                    options=[30, 60, 90, 180, 360],  # minutes
                    index=2,  # Default to 90 minutes
                    format_func=lambda x: f"{x} minutes" if x < 60 else f"{x//60} hours"
                )
                
                # Add a refresh button for manual updates
                if st.sidebar.button("🔄 Update Positions"):
                    # Advance time by selected step for visible movement
                    st.session_state.time_offset += time_step * 60  # Convert to seconds
                    st.rerun()
                
                # Add reset button
                if st.sidebar.button("🔄 Reset Positions"):
                    st.session_state.time_offset = 0
                    st.rerun()
                
                # Show movement status with orbital progress
                hours_passed = st.session_state.time_offset / 3600
                iss_orbits = hours_passed / 1.55  # ISS orbital period ~93 minutes
                st.sidebar.info(f"💫 Time advanced: {hours_passed:.1f} hours\n🛰️ ISS completed: {iss_orbits:.1f} orbits")
            
            # No loading spinner for faster performance
            
            self.render_metrics(controls)
            
            # Main tabs
            tab1, tab2, tab3, tab4 = st.tabs([
                "🌍 3D Orbital View",
                "🗺️ Ground Network", 
                "📊 Risk Analysis",
                "🚨 System Status"
            ])
            
            with tab1:
                st.subheader("3D Orbital Environment")
                
                # Show active filters
                active_filters = []
                if controls['show_debris']:
                    active_filters.append("Space Debris")
                if controls['show_satellites']:
                    active_filters.append("Satellites")
                if controls['show_ground_stations']:
                    active_filters.append("Ground Stations")
                
                orbital_filters = []
                if controls['show_leo']:
                    orbital_filters.append("LEO")
                if controls['show_meo']:
                    orbital_filters.append("MEO") 
                if controls['show_geo']:
                    orbital_filters.append("GEO")
                
                if active_filters:
                    st.info(f"🔍 **Active Filters:** {', '.join(active_filters)} | **Orbits:** {', '.join(orbital_filters)} | **Collision Threshold:** {controls['collision_threshold']}km")
                else:
                    st.warning("⚠️ **No objects selected for display**")
                
                fig_3d = self.create_3d_orbital_view(controls)
                st.plotly_chart(fig_3d, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    # Dynamic info based on filters
                    if controls['show_debris']:
                        st.info("🛰️ Space debris objects visible")
                    else:
                        st.warning("🛰️ Space debris objects hidden")
                    
                    if controls['show_satellites']:
                        st.info("📡 Active satellites visible")
                    else:
                        st.warning("📡 Active satellites hidden")
                        
                with col2:
                    if controls['show_ground_stations']:
                        st.info("🏢 Ground stations visible")
                    else:
                        st.warning("🏢 Ground stations hidden")
                        
                    st.success(f"🎯 Collision threshold: {controls['collision_threshold']} km")
            
            with tab2:
                st.subheader("Global Ground Station Network")
                fig_map = self.create_ground_stations_chart()
                st.plotly_chart(fig_map, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Station Status**")
                    st.success("🟢 Goldstone: Online")
                    st.success("🟢 Canberra: Online") 
                    st.success("🟢 Madrid: Online")
                with col2:
                    st.write("**Network Status**")
                    st.warning("🟡 Kourou: Maintenance")
                    st.success("🟢 Svalbard: Online")
                    st.success("🟢 Hawaii: Online")
            
            with tab3:
                st.subheader("Collision Risk Analysis")
                fig_risk = self.create_collision_risk_chart()
                st.plotly_chart(fig_risk, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("High Risk Objects", "5", "-2")
                    st.metric("Medium Risk Objects", "18", "+1")
                with col2:
                    st.metric("Collision Probability", "0.001%", "-0.002%")
                    st.metric("Next Close Approach", "2.3 hours", "")
            
            with tab4:
                st.subheader("System Health & Performance")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Data Update Rate", "99.8%", "+0.2%")
                    st.metric("Tracking Accuracy", "±50m", "±5m")
                
                with col2:
                    st.metric("Dashboard Response", "45ms", "-5ms")
                    st.metric("Active Connections", "67", "+3")
                
                with col3:
                    st.metric("Alert Processing", "Real-time", "")
                    st.metric("System Uptime", "99.99%", "")
                
                st.success("🟢 All systems operational")
                st.info("🔄 Next TLE update in 2 hours")
                st.info("📊 Dashboard running smoothly")
                
        except Exception as e:
            st.error(f"Dashboard error: {str(e)}")
            logger.error(f"Dashboard error: {e}", exc_info=True)


def main():
    """Main function."""
    dashboard = MinimalDebrisVisualizationDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
