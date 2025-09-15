#!/usr/bin/env python3
"""
Space Debris Tracking Module

This module handles the tracking and monitoring of space debris objects,
including TLE data management, orbital propagation, and collision detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import sys
import requests
from skyfield.api import load, wgs84
from skyfield.timelib import Time
import sqlite3
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.leo_simulation.geometry import SatelliteGeometry

logger = logging.getLogger(__name__)


class DebrisTracker:
    """
    Advanced space debris tracking system with real-time monitoring,
    collision detection, and risk assessment capabilities.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize the debris tracker with database and TLE sources."""
        self.db_path = db_path or "debris_tracking.db"
        self.satellite_geometry = SatelliteGeometry()
        
        # TLE data sources for different object types
        self.tle_sources = {
            'active_satellites': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle',
            'debris': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=tle',
            'rocket_bodies': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=rocket-bodies&FORMAT=tle',
            'analyst_satellites': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=analyst&FORMAT=tle',
            'cubesats': 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cubesat&FORMAT=tle'
        }
        
        # Initialize database
        self.init_database()
        
        # Collision detection parameters
        self.collision_threshold_km = 50.0  # Alert threshold
        self.critical_threshold_km = 5.0    # Critical alert threshold
        
        logger.info("DebrisTracker initialized successfully")
    
    def init_database(self):
        """Initialize SQLite database for debris tracking data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables for debris objects
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS debris_objects (
                    norad_id INTEGER PRIMARY KEY,
                    object_name TEXT NOT NULL,
                    object_type TEXT,
                    launch_date TEXT,
                    decay_date TEXT,
                    rcs_size TEXT,
                    country TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create table for orbital elements
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS orbital_elements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id INTEGER,
                    epoch TIMESTAMP,
                    mean_motion REAL,
                    eccentricity REAL,
                    inclination REAL,
                    raan REAL,
                    arg_perigee REAL,
                    mean_anomaly REAL,
                    bstar REAL,
                    tle_line1 TEXT,
                    tle_line2 TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (norad_id) REFERENCES debris_objects (norad_id)
                )
            ''')
            
            # Create table for collision events
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS collision_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object1_norad_id INTEGER,
                    object2_norad_id INTEGER,
                    min_distance_km REAL,
                    collision_probability REAL,
                    time_of_closest_approach TIMESTAMP,
                    alert_level TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create table for tracking history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tracking_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    norad_id INTEGER,
                    timestamp TIMESTAMP,
                    position_x_km REAL,
                    position_y_km REAL,
                    position_z_km REAL,
                    velocity_x_kms REAL,
                    velocity_y_kms REAL,
                    velocity_z_kms REAL,
                    altitude_km REAL,
                    latitude REAL,
                    longitude REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def update_debris_catalog(self, source_types: List[str] = None) -> Dict[str, int]:
        """
        Update the debris catalog by downloading fresh TLE data.
        
        Args:
            source_types: List of source types to update. If None, updates all.
            
        Returns:
            Dictionary with counts of objects updated per source type.
        """
        if source_types is None:
            source_types = list(self.tle_sources.keys())
        
        update_counts = {}
        
        for source_type in source_types:
            try:
                logger.info(f"Updating {source_type} catalog...")
                
                # Download TLE data
                tle_url = self.tle_sources[source_type]
                filename = f"{source_type}.tle"
                
                # Load TLE data using satellite geometry
                satellites = self.satellite_geometry.load_tle_data(
                    tle_url=tle_url,
                    filename=filename,
                    max_age_days=0.5  # Update twice daily
                )
                
                # Process and store in database
                count = self._store_tle_data(satellites, source_type)
                update_counts[source_type] = count
                
                logger.info(f"Updated {count} objects from {source_type}")
                
            except Exception as e:
                logger.error(f"Failed to update {source_type}: {e}")
                update_counts[source_type] = 0
        
        return update_counts
    
    def _store_tle_data(self, satellites: Dict[str, Any], object_type: str) -> int:
        """Store TLE data in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stored_count = 0
            
            for sat_name, satellite in satellites.items():
                try:
                    norad_id = satellite.model.satnum
                    
                    # Extract orbital elements
                    tle_lines = str(satellite).split('\n')
                    if len(tle_lines) >= 3:
                        tle_line1 = tle_lines[1].strip()
                        tle_line2 = tle_lines[2].strip()
                        
                        # Parse orbital elements from TLE
                        epoch = satellite.epoch.utc_datetime()
                        mean_motion = satellite.model.no_kozai * 1440 / (2 * np.pi)  # rev/day
                        eccentricity = satellite.model.ecco
                        inclination = np.degrees(satellite.model.inclo)
                        raan = np.degrees(satellite.model.nodeo)
                        arg_perigee = np.degrees(satellite.model.argpo)
                        mean_anomaly = np.degrees(satellite.model.mo)
                        bstar = satellite.model.bstar
                        
                        # Insert or update debris object
                        cursor.execute('''
                            INSERT OR REPLACE INTO debris_objects 
                            (norad_id, object_name, object_type, last_updated)
                            VALUES (?, ?, ?, ?)
                        ''', (norad_id, sat_name, object_type, datetime.now()))
                        
                        # Insert orbital elements
                        cursor.execute('''
                            INSERT INTO orbital_elements 
                            (norad_id, epoch, mean_motion, eccentricity, inclination, 
                             raan, arg_perigee, mean_anomaly, bstar, tle_line1, tle_line2)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (norad_id, epoch, mean_motion, eccentricity, inclination,
                              raan, arg_perigee, mean_anomaly, bstar, tle_line1, tle_line2))
                        
                        stored_count += 1
                        
                except Exception as e:
                    logger.warning(f"Failed to store satellite {sat_name}: {e}")
                    continue
            
            conn.commit()
            conn.close()
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Database storage failed: {e}")
            raise
    
    def get_tracked_objects(self, object_types: List[str] = None, 
                           limit: int = None) -> pd.DataFrame:
        """
        Get tracked objects from the database.
        
        Args:
            object_types: Filter by object types
            limit: Maximum number of objects to return
            
        Returns:
            DataFrame with tracked objects information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT d.norad_id, d.object_name, d.object_type, d.last_updated,
                       oe.epoch, oe.mean_motion, oe.eccentricity, oe.inclination,
                       oe.raan, oe.arg_perigee, oe.mean_anomaly, oe.bstar,
                       oe.tle_line1, oe.tle_line2
                FROM debris_objects d
                LEFT JOIN orbital_elements oe ON d.norad_id = oe.norad_id
                WHERE oe.id IN (
                    SELECT MAX(id) FROM orbital_elements GROUP BY norad_id
                )
            '''
            
            params = []
            if object_types:
                placeholders = ','.join(['?' for _ in object_types])
                query += f' AND d.object_type IN ({placeholders})'
                params.extend(object_types)
            
            if limit:
                query += ' LIMIT ?'
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get tracked objects: {e}")
            return pd.DataFrame()
    
    def calculate_current_positions(self, norad_ids: List[int] = None,
                                   timestamp: datetime = None) -> pd.DataFrame:
        """
        Calculate current positions of tracked objects.
        
        Args:
            norad_ids: Specific objects to calculate positions for
            timestamp: Time for position calculation (default: now)
            
        Returns:
            DataFrame with object positions
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        try:
            # Get objects to process
            if norad_ids:
                objects_df = self.get_tracked_objects()
                objects_df = objects_df[objects_df['norad_id'].isin(norad_ids)]
            else:
                objects_df = self.get_tracked_objects(limit=1000)  # Limit for performance
            
            if objects_df.empty:
                return pd.DataFrame()
            
            positions = []
            ts = self.satellite_geometry.ts
            t = ts.from_datetime(timestamp)
            
            for _, obj in objects_df.iterrows():
                try:
                    # Create satellite object from TLE
                    if pd.notna(obj['tle_line1']) and pd.notna(obj['tle_line2']):
                        lines = [obj['object_name'], obj['tle_line1'], obj['tle_line2']]
                        satellite = self.satellite_geometry.loader.tle_file_from_lines(lines)[0]
                        
                        # Calculate position
                        geocentric = satellite.at(t)
                        position = geocentric.position.km
                        velocity = geocentric.velocity.km_per_s
                        
                        # Calculate altitude and lat/lon
                        subpoint = geocentric.subpoint()
                        altitude_km = subpoint.elevation.km
                        latitude = subpoint.latitude.degrees
                        longitude = subpoint.longitude.degrees
                        
                        positions.append({
                            'norad_id': obj['norad_id'],
                            'object_name': obj['object_name'],
                            'object_type': obj['object_type'],
                            'timestamp': timestamp,
                            'x_km': position[0],
                            'y_km': position[1],
                            'z_km': position[2],
                            'vx_kms': velocity[0],
                            'vy_kms': velocity[1],
                            'vz_kms': velocity[2],
                            'altitude_km': altitude_km,
                            'latitude': latitude,
                            'longitude': longitude
                        })
                        
                except Exception as e:
                    logger.warning(f"Failed to calculate position for {obj['object_name']}: {e}")
                    continue
            
            return pd.DataFrame(positions)
            
        except Exception as e:
            logger.error(f"Position calculation failed: {e}")
            return pd.DataFrame()
    
    def detect_collisions(self, time_window_hours: int = 24,
                         threshold_km: float = None) -> pd.DataFrame:
        """
        Detect potential collisions between tracked objects.
        
        Args:
            time_window_hours: Time window to check for collisions
            threshold_km: Distance threshold for collision alerts
            
        Returns:
            DataFrame with potential collision events
        """
        if threshold_km is None:
            threshold_km = self.collision_threshold_km
        
        try:
            logger.info(f"Detecting collisions over {time_window_hours}h window...")
            
            # Get current positions of all objects
            current_positions = self.calculate_current_positions()
            
            if len(current_positions) < 2:
                return pd.DataFrame()
            
            collision_events = []
            
            # Check all pairs of objects
            for i in range(len(current_positions)):
                for j in range(i + 1, len(current_positions)):
                    obj1 = current_positions.iloc[i]
                    obj2 = current_positions.iloc[j]
                    
                    # Calculate distance
                    dx = obj1['x_km'] - obj2['x_km']
                    dy = obj1['y_km'] - obj2['y_km']
                    dz = obj1['z_km'] - obj2['z_km']
                    distance = np.sqrt(dx**2 + dy**2 + dz**2)
                    
                    if distance < threshold_km:
                        # Calculate relative velocity
                        dvx = obj1['vx_kms'] - obj2['vx_kms']
                        dvy = obj1['vy_kms'] - obj2['vy_kms']
                        dvz = obj1['vz_kms'] - obj2['vz_kms']
                        rel_velocity = np.sqrt(dvx**2 + dvy**2 + dvz**2)
                        
                        # Estimate collision probability (simplified)
                        collision_prob = self._calculate_collision_probability(
                            distance, rel_velocity, obj1, obj2
                        )
                        
                        # Determine alert level
                        if distance < self.critical_threshold_km:
                            alert_level = 'CRITICAL'
                        elif distance < threshold_km * 0.5:
                            alert_level = 'HIGH'
                        elif distance < threshold_km * 0.8:
                            alert_level = 'MEDIUM'
                        else:
                            alert_level = 'LOW'
                        
                        collision_events.append({
                            'object1_norad_id': obj1['norad_id'],
                            'object1_name': obj1['object_name'],
                            'object2_norad_id': obj2['norad_id'],
                            'object2_name': obj2['object_name'],
                            'distance_km': distance,
                            'relative_velocity_kms': rel_velocity,
                            'collision_probability': collision_prob,
                            'alert_level': alert_level,
                            'detection_time': datetime.now()
                        })
            
            collision_df = pd.DataFrame(collision_events)
            
            # Store collision events in database
            if not collision_df.empty:
                self._store_collision_events(collision_df)
            
            logger.info(f"Detected {len(collision_events)} potential collision events")
            return collision_df
            
        except Exception as e:
            logger.error(f"Collision detection failed: {e}")
            return pd.DataFrame()
    
    def _calculate_collision_probability(self, distance_km: float, 
                                       rel_velocity_kms: float,
                                       obj1: pd.Series, obj2: pd.Series) -> float:
        """
        Calculate collision probability using simplified model.
        
        This is a basic implementation. In practice, you'd use more sophisticated
        models like PC (Probability of Collision) calculations.
        """
        try:
            # Simplified collision probability model
            # Based on distance, relative velocity, and object sizes
            
            # Estimate cross-sectional areas (very rough approximation)
            # In reality, you'd use radar cross-section data
            area1 = 10.0  # m²
            area2 = 10.0  # m²
            combined_area = np.sqrt(area1 + area2)
            
            # Convert distance to meters
            distance_m = distance_km * 1000
            
            # Basic probability calculation
            if distance_m <= 0:
                return 1.0
            
            # Probability decreases with distance, increases with combined cross-section
            base_prob = (combined_area / (np.pi * distance_m**2))
            
            # Adjust for relative velocity (higher velocity = less time to maneuver)
            velocity_factor = min(rel_velocity_kms / 10.0, 2.0)  # Cap at 2x
            
            probability = base_prob * velocity_factor
            
            # Cap probability at 1.0
            return min(probability, 1.0)
            
        except Exception as e:
            logger.warning(f"Collision probability calculation failed: {e}")
            return 0.0
    
    def _store_collision_events(self, collision_df: pd.DataFrame):
        """Store collision events in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            for _, event in collision_df.iterrows():
                conn.execute('''
                    INSERT INTO collision_events 
                    (object1_norad_id, object2_norad_id, min_distance_km, 
                     collision_probability, alert_level)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    event['object1_norad_id'],
                    event['object2_norad_id'],
                    event['distance_km'],
                    event['collision_probability'],
                    event['alert_level']
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store collision events: {e}")
    
    def get_collision_history(self, hours_back: int = 24) -> pd.DataFrame:
        """Get collision event history from the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = '''
                SELECT ce.*, 
                       d1.object_name as object1_name,
                       d2.object_name as object2_name
                FROM collision_events ce
                LEFT JOIN debris_objects d1 ON ce.object1_norad_id = d1.norad_id
                LEFT JOIN debris_objects d2 ON ce.object2_norad_id = d2.norad_id
                WHERE ce.created_at > datetime('now', '-{} hours')
                ORDER BY ce.created_at DESC
            '''.format(hours_back)
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get collision history: {e}")
            return pd.DataFrame()
    
    def generate_alerts(self, collision_df: pd.DataFrame = None) -> List[Dict]:
        """
        Generate alerts based on collision detection results.
        
        Args:
            collision_df: Collision detection results. If None, runs detection.
            
        Returns:
            List of alert dictionaries
        """
        try:
            if collision_df is None:
                collision_df = self.detect_collisions()
            
            alerts = []
            
            for _, event in collision_df.iterrows():
                severity_map = {
                    'CRITICAL': 4,
                    'HIGH': 3,
                    'MEDIUM': 2,
                    'LOW': 1
                }
                
                alert = {
                    'type': 'Collision Risk',
                    'message': f"{event['object1_name']} and {event['object2_name']} "
                              f"within {event['distance_km']:.1f}km "
                              f"(P={event['collision_probability']:.3f})",
                    'severity': severity_map.get(event['alert_level'], 1),
                    'alert_level': event['alert_level'],
                    'timestamp': datetime.now(),
                    'objects_involved': [
                        event['object1_norad_id'],
                        event['object2_norad_id']
                    ],
                    'distance_km': event['distance_km'],
                    'collision_probability': event['collision_probability']
                }
                
                alerts.append(alert)
            
            # Sort by severity (highest first)
            alerts.sort(key=lambda x: x['severity'], reverse=True)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Alert generation failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracking system statistics."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            stats = {}
            
            # Object counts by type
            cursor = conn.execute('''
                SELECT object_type, COUNT(*) as count
                FROM debris_objects
                GROUP BY object_type
            ''')
            stats['object_counts'] = dict(cursor.fetchall())
            
            # Total objects
            cursor = conn.execute('SELECT COUNT(*) FROM debris_objects')
            stats['total_objects'] = cursor.fetchone()[0]
            
            # Recent collision events
            cursor = conn.execute('''
                SELECT alert_level, COUNT(*) as count
                FROM collision_events
                WHERE created_at > datetime('now', '-24 hours')
                GROUP BY alert_level
            ''')
            stats['recent_collision_alerts'] = dict(cursor.fetchall())
            
            # Last update time
            cursor = conn.execute('''
                SELECT MAX(last_updated) FROM debris_objects
            ''')
            last_update = cursor.fetchone()[0]
            stats['last_catalog_update'] = last_update
            
            conn.close()
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}


def main():
    """Main function for testing the debris tracker."""
    logging.basicConfig(level=logging.INFO)
    
    tracker = DebrisTracker()
    
    # Update catalog
    print("Updating debris catalog...")
    update_counts = tracker.update_debris_catalog(['debris', 'active_satellites'])
    print(f"Update results: {update_counts}")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"Tracking statistics: {stats}")
    
    # Detect collisions
    print("Detecting collisions...")
    collisions = tracker.detect_collisions()
    print(f"Found {len(collisions)} potential collision events")
    
    # Generate alerts
    alerts = tracker.generate_alerts(collisions)
    print(f"Generated {len(alerts)} alerts")


if __name__ == "__main__":
    main()
