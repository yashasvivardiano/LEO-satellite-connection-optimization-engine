#!/usr/bin/env python3
"""
Network Topology Module

This module handles the visualization and management of wired network
infrastructure, ground station networks, and their interconnections.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import networkx as nx
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class NetworkTopology:
    """
    Network topology manager for ground stations and wired infrastructure.
    
    Handles:
    - Ground station network topology
    - Wired network connections (fiber, satellite links, etc.)
    - Network performance monitoring
    - Redundancy and failover analysis
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the network topology manager."""
        self.config_path = config_path or "config/network_topology.json"
        self.network_graph = nx.Graph()
        
        # Network configuration
        self.ground_stations = {}
        self.wired_connections = {}
        self.network_performance = {}
        
        # Load configuration if available
        self.load_network_config()
        
        logger.info("NetworkTopology initialized successfully")
    
    def load_network_config(self):
        """Load network configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.ground_stations = config.get('ground_stations', {})
                self.wired_connections = config.get('wired_connections', {})
                
                # Build network graph
                self._build_network_graph()
                
                logger.info(f"Loaded network config with {len(self.ground_stations)} stations")
            else:
                logger.info("No network config found, using defaults")
                self._create_default_network()
                
        except Exception as e:
            logger.error(f"Failed to load network config: {e}")
            self._create_default_network()
    
    def _create_default_network(self):
        """Create a default network topology for demonstration."""
        # Default ground stations (major space communication facilities)
        default_stations = {
            'goldstone': {
                'name': 'Goldstone Deep Space Communications Complex',
                'location': {'lat': 35.426667, 'lon': -116.890000},
                'type': 'deep_space',
                'antennas': ['DSS-14', 'DSS-24', 'DSS-25'],
                'capabilities': ['deep_space', 'leo', 'geo'],
                'status': 'operational',
                'capacity_gbps': 100
            },
            'canberra': {
                'name': 'Canberra Deep Space Communication Complex',
                'location': {'lat': -35.401389, 'lon': 148.981667},
                'type': 'deep_space',
                'antennas': ['DSS-34', 'DSS-35', 'DSS-36'],
                'capabilities': ['deep_space', 'leo', 'geo'],
                'status': 'operational',
                'capacity_gbps': 100
            },
            'madrid': {
                'name': 'Madrid Deep Space Communication Complex',
                'location': {'lat': 40.427222, 'lon': -4.248333},
                'type': 'deep_space',
                'antennas': ['DSS-54', 'DSS-55', 'DSS-56'],
                'capabilities': ['deep_space', 'leo', 'geo'],
                'status': 'operational',
                'capacity_gbps': 100
            },
            'kourou': {
                'name': 'Kourou Space Center',
                'location': {'lat': 5.239444, 'lon': -52.768333},
                'type': 'launch_control',
                'antennas': ['KOU-1', 'KOU-2'],
                'capabilities': ['leo', 'geo', 'launch_control'],
                'status': 'operational',
                'capacity_gbps': 50
            },
            'svalbard': {
                'name': 'Svalbard Satellite Station',
                'location': {'lat': 78.229722, 'lon': 15.407778},
                'type': 'polar_orbit',
                'antennas': ['SG1', 'SG2'],
                'capabilities': ['leo', 'polar_orbit'],
                'status': 'operational',
                'capacity_gbps': 25
            },
            'hawaii': {
                'name': 'Hawaii Tracking Station',
                'location': {'lat': 21.561944, 'lon': -158.066667},
                'type': 'tracking',
                'antennas': ['HAW-1'],
                'capabilities': ['leo', 'geo', 'tracking'],
                'status': 'operational',
                'capacity_gbps': 30
            }
        }
        
        # Default wired connections
        default_connections = {
            'goldstone_canberra': {
                'endpoints': ['goldstone', 'canberra'],
                'type': 'fiber_optic',
                'capacity_gbps': 40,
                'latency_ms': 180,
                'redundancy': 'primary',
                'status': 'operational'
            },
            'canberra_madrid': {
                'endpoints': ['canberra', 'madrid'],
                'type': 'fiber_optic',
                'capacity_gbps': 40,
                'latency_ms': 220,
                'redundancy': 'primary',
                'status': 'operational'
            },
            'madrid_kourou': {
                'endpoints': ['madrid', 'kourou'],
                'type': 'satellite_link',
                'capacity_gbps': 10,
                'latency_ms': 250,
                'redundancy': 'backup',
                'status': 'operational'
            },
            'kourou_hawaii': {
                'endpoints': ['kourou', 'hawaii'],
                'type': 'undersea_cable',
                'capacity_gbps': 20,
                'latency_ms': 150,
                'redundancy': 'primary',
                'status': 'operational'
            },
            'hawaii_goldstone': {
                'endpoints': ['hawaii', 'goldstone'],
                'type': 'fiber_optic',
                'capacity_gbps': 30,
                'latency_ms': 50,
                'redundancy': 'primary',
                'status': 'operational'
            },
            'svalbard_madrid': {
                'endpoints': ['svalbard', 'madrid'],
                'type': 'satellite_link',
                'capacity_gbps': 15,
                'latency_ms': 300,
                'redundancy': 'backup',
                'status': 'operational'
            }
        }
        
        self.ground_stations = default_stations
        self.wired_connections = default_connections
        
        # Build network graph
        self._build_network_graph()
        
        # Save default configuration
        self.save_network_config()
    
    def _build_network_graph(self):
        """Build NetworkX graph from station and connection data."""
        self.network_graph.clear()
        
        # Add ground stations as nodes
        for station_id, station_data in self.ground_stations.items():
            self.network_graph.add_node(
                station_id,
                **station_data
            )
        
        # Add connections as edges
        for conn_id, conn_data in self.wired_connections.items():
            endpoints = conn_data['endpoints']
            if len(endpoints) == 2:
                self.network_graph.add_edge(
                    endpoints[0],
                    endpoints[1],
                    connection_id=conn_id,
                    **{k: v for k, v in conn_data.items() if k != 'endpoints'}
                )
    
    def save_network_config(self):
        """Save current network configuration to file."""
        try:
            config = {
                'ground_stations': self.ground_stations,
                'wired_connections': self.wired_connections,
                'last_updated': datetime.now().isoformat()
            }
            
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Network configuration saved to {self.config_path}")
            
        except Exception as e:
            logger.error(f"Failed to save network config: {e}")
    
    def get_ground_stations_df(self) -> pd.DataFrame:
        """Get ground stations as a pandas DataFrame."""
        try:
            stations_list = []
            
            for station_id, station_data in self.ground_stations.items():
                station_record = {
                    'station_id': station_id,
                    'station_name': station_data.get('name', station_id),
                    'latitude': station_data['location']['lat'],
                    'longitude': station_data['location']['lon'],
                    'station_type': station_data.get('type', 'unknown'),
                    'status': station_data.get('status', 'unknown'),
                    'capacity_gbps': station_data.get('capacity_gbps', 0),
                    'antenna_count': len(station_data.get('antennas', [])),
                    'capabilities': ', '.join(station_data.get('capabilities', []))
                }
                stations_list.append(station_record)
            
            return pd.DataFrame(stations_list)
            
        except Exception as e:
            logger.error(f"Failed to create stations DataFrame: {e}")
            return pd.DataFrame()
    
    def get_wired_connections_df(self) -> pd.DataFrame:
        """Get wired connections as a pandas DataFrame."""
        try:
            connections_list = []
            
            for conn_id, conn_data in self.wired_connections.items():
                endpoints = conn_data.get('endpoints', [])
                if len(endpoints) >= 2:
                    # Get endpoint coordinates
                    start_station = self.ground_stations.get(endpoints[0], {})
                    end_station = self.ground_stations.get(endpoints[1], {})
                    
                    start_loc = start_station.get('location', {})
                    end_loc = end_station.get('location', {})
                    
                    connection_record = {
                        'connection_id': conn_id,
                        'connection_type': conn_data.get('type', 'unknown'),
                        'start_station': endpoints[0],
                        'end_station': endpoints[1],
                        'start_lat': start_loc.get('lat', 0),
                        'start_lon': start_loc.get('lon', 0),
                        'end_lat': end_loc.get('lat', 0),
                        'end_lon': end_loc.get('lon', 0),
                        'capacity_gbps': conn_data.get('capacity_gbps', 0),
                        'latency_ms': conn_data.get('latency_ms', 0),
                        'redundancy': conn_data.get('redundancy', 'unknown'),
                        'status': conn_data.get('status', 'unknown'),
                        'utilization_percent': np.random.randint(30, 90)  # Simulated
                    }
                    connections_list.append(connection_record)
            
            return pd.DataFrame(connections_list)
            
        except Exception as e:
            logger.error(f"Failed to create connections DataFrame: {e}")
            return pd.DataFrame()
    
    def analyze_network_redundancy(self) -> Dict[str, Any]:
        """Analyze network redundancy and single points of failure."""
        try:
            analysis = {
                'total_nodes': self.network_graph.number_of_nodes(),
                'total_edges': self.network_graph.number_of_edges(),
                'connectivity': {},
                'critical_nodes': [],
                'critical_edges': [],
                'redundant_paths': {}
            }
            
            # Check overall connectivity
            analysis['connectivity']['is_connected'] = nx.is_connected(self.network_graph)
            
            if nx.is_connected(self.network_graph):
                analysis['connectivity']['diameter'] = nx.diameter(self.network_graph)
                analysis['connectivity']['average_shortest_path'] = nx.average_shortest_path_length(
                    self.network_graph
                )
            
            # Find articulation points (critical nodes)
            articulation_points = list(nx.articulation_points(self.network_graph))
            analysis['critical_nodes'] = articulation_points
            
            # Find bridges (critical edges)
            bridges = list(nx.bridges(self.network_graph))
            analysis['critical_edges'] = bridges
            
            # Analyze redundant paths between major nodes
            major_nodes = [node for node, data in self.network_graph.nodes(data=True)
                          if data.get('type') == 'deep_space']
            
            for i, node1 in enumerate(major_nodes):
                for node2 in major_nodes[i+1:]:
                    try:
                        # Find all simple paths
                        paths = list(nx.all_simple_paths(
                            self.network_graph, node1, node2, cutoff=4
                        ))
                        analysis['redundant_paths'][f"{node1}_{node2}"] = {
                            'path_count': len(paths),
                            'shortest_path_length': nx.shortest_path_length(
                                self.network_graph, node1, node2
                            ),
                            'paths': paths[:3]  # Store first 3 paths
                        }
                    except nx.NetworkXNoPath:
                        analysis['redundant_paths'][f"{node1}_{node2}"] = {
                            'path_count': 0,
                            'shortest_path_length': float('inf'),
                            'paths': []
                        }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Network redundancy analysis failed: {e}")
            return {}
    
    def simulate_network_failure(self, failed_nodes: List[str] = None,
                                failed_edges: List[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Simulate network failures and analyze impact.
        
        Args:
            failed_nodes: List of station IDs to simulate as failed
            failed_edges: List of connection tuples to simulate as failed
            
        Returns:
            Analysis of network state after failures
        """
        try:
            # Create a copy of the network for simulation
            sim_graph = self.network_graph.copy()
            
            # Remove failed nodes
            if failed_nodes:
                for node in failed_nodes:
                    if node in sim_graph:
                        sim_graph.remove_node(node)
            
            # Remove failed edges
            if failed_edges:
                for edge in failed_edges:
                    if sim_graph.has_edge(edge[0], edge[1]):
                        sim_graph.remove_edge(edge[0], edge[1])
            
            # Analyze the degraded network
            analysis = {
                'original_nodes': self.network_graph.number_of_nodes(),
                'remaining_nodes': sim_graph.number_of_nodes(),
                'original_edges': self.network_graph.number_of_edges(),
                'remaining_edges': sim_graph.number_of_edges(),
                'failed_nodes': failed_nodes or [],
                'failed_edges': failed_edges or [],
                'connectivity_impact': {},
                'isolated_nodes': [],
                'affected_capabilities': []
            }
            
            # Check connectivity
            if sim_graph.number_of_nodes() > 0:
                analysis['connectivity_impact']['is_connected'] = nx.is_connected(sim_graph)
                
                # Find isolated nodes
                if not nx.is_connected(sim_graph):
                    components = list(nx.connected_components(sim_graph))
                    largest_component = max(components, key=len)
                    
                    for component in components:
                        if component != largest_component:
                            analysis['isolated_nodes'].extend(list(component))
                
                # Calculate degraded performance
                if nx.is_connected(sim_graph) and sim_graph.number_of_nodes() > 1:
                    try:
                        analysis['connectivity_impact']['diameter'] = nx.diameter(sim_graph)
                        analysis['connectivity_impact']['avg_path_length'] = \
                            nx.average_shortest_path_length(sim_graph)
                    except:
                        pass
            
            # Analyze capability impact
            for node in (failed_nodes or []):
                if node in self.ground_stations:
                    station_capabilities = self.ground_stations[node].get('capabilities', [])
                    analysis['affected_capabilities'].extend(station_capabilities)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Network failure simulation failed: {e}")
            return {}
    
    def get_network_performance_metrics(self) -> Dict[str, Any]:
        """Get current network performance metrics."""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'overall_health': 'healthy',
                'station_metrics': {},
                'connection_metrics': {},
                'aggregate_metrics': {
                    'total_capacity_gbps': 0,
                    'average_utilization': 0,
                    'average_latency_ms': 0,
                    'operational_stations': 0,
                    'operational_connections': 0
                }
            }
            
            # Station metrics
            operational_stations = 0
            for station_id, station_data in self.ground_stations.items():
                status = station_data.get('status', 'unknown')
                capacity = station_data.get('capacity_gbps', 0)
                
                # Simulate performance metrics
                utilization = np.random.uniform(40, 85) if status == 'operational' else 0
                response_time = np.random.uniform(10, 50) if status == 'operational' else 1000
                
                metrics['station_metrics'][station_id] = {
                    'status': status,
                    'capacity_gbps': capacity,
                    'utilization_percent': utilization,
                    'response_time_ms': response_time,
                    'antenna_count': len(station_data.get('antennas', []))
                }
                
                if status == 'operational':
                    operational_stations += 1
                    metrics['aggregate_metrics']['total_capacity_gbps'] += capacity
            
            # Connection metrics
            operational_connections = 0
            total_latency = 0
            connection_count = 0
            
            for conn_id, conn_data in self.wired_connections.items():
                status = conn_data.get('status', 'unknown')
                capacity = conn_data.get('capacity_gbps', 0)
                latency = conn_data.get('latency_ms', 0)
                
                # Simulate performance metrics
                utilization = np.random.uniform(30, 80) if status == 'operational' else 0
                packet_loss = np.random.uniform(0, 0.1) if status == 'operational' else 100
                
                metrics['connection_metrics'][conn_id] = {
                    'status': status,
                    'capacity_gbps': capacity,
                    'utilization_percent': utilization,
                    'latency_ms': latency,
                    'packet_loss_percent': packet_loss,
                    'type': conn_data.get('type', 'unknown')
                }
                
                if status == 'operational':
                    operational_connections += 1
                    total_latency += latency
                    connection_count += 1
            
            # Calculate aggregate metrics
            metrics['aggregate_metrics']['operational_stations'] = operational_stations
            metrics['aggregate_metrics']['operational_connections'] = operational_connections
            
            if connection_count > 0:
                metrics['aggregate_metrics']['average_latency_ms'] = total_latency / connection_count
            
            # Calculate average utilization across all operational components
            all_utilizations = []
            for station_metrics in metrics['station_metrics'].values():
                if station_metrics['status'] == 'operational':
                    all_utilizations.append(station_metrics['utilization_percent'])
            
            for conn_metrics in metrics['connection_metrics'].values():
                if conn_metrics['status'] == 'operational':
                    all_utilizations.append(conn_metrics['utilization_percent'])
            
            if all_utilizations:
                metrics['aggregate_metrics']['average_utilization'] = np.mean(all_utilizations)
            
            # Determine overall health
            if operational_stations < len(self.ground_stations) * 0.8:
                metrics['overall_health'] = 'degraded'
            elif operational_connections < len(self.wired_connections) * 0.7:
                metrics['overall_health'] = 'degraded'
            elif metrics['aggregate_metrics']['average_utilization'] > 90:
                metrics['overall_health'] = 'congested'
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get network performance metrics: {e}")
            return {}
    
    def add_ground_station(self, station_id: str, station_data: Dict[str, Any]):
        """Add a new ground station to the network."""
        try:
            self.ground_stations[station_id] = station_data
            self._build_network_graph()
            self.save_network_config()
            
            logger.info(f"Added ground station: {station_id}")
            
        except Exception as e:
            logger.error(f"Failed to add ground station {station_id}: {e}")
    
    def add_wired_connection(self, connection_id: str, connection_data: Dict[str, Any]):
        """Add a new wired connection to the network."""
        try:
            self.wired_connections[connection_id] = connection_data
            self._build_network_graph()
            self.save_network_config()
            
            logger.info(f"Added wired connection: {connection_id}")
            
        except Exception as e:
            logger.error(f"Failed to add wired connection {connection_id}: {e}")
    
    def update_station_status(self, station_id: str, status: str):
        """Update the status of a ground station."""
        try:
            if station_id in self.ground_stations:
                self.ground_stations[station_id]['status'] = status
                self._build_network_graph()
                
                logger.info(f"Updated station {station_id} status to {status}")
            else:
                logger.warning(f"Station {station_id} not found")
                
        except Exception as e:
            logger.error(f"Failed to update station status: {e}")
    
    def update_connection_status(self, connection_id: str, status: str):
        """Update the status of a wired connection."""
        try:
            if connection_id in self.wired_connections:
                self.wired_connections[connection_id]['status'] = status
                self._build_network_graph()
                
                logger.info(f"Updated connection {connection_id} status to {status}")
            else:
                logger.warning(f"Connection {connection_id} not found")
                
        except Exception as e:
            logger.error(f"Failed to update connection status: {e}")


def main():
    """Main function for testing the network topology module."""
    logging.basicConfig(level=logging.INFO)
    
    network = NetworkTopology()
    
    # Get network data
    stations_df = network.get_ground_stations_df()
    connections_df = network.get_wired_connections_df()
    
    print(f"Ground Stations: {len(stations_df)}")
    print(stations_df[['station_name', 'station_type', 'status']].to_string())
    
    print(f"\nWired Connections: {len(connections_df)}")
    print(connections_df[['connection_type', 'start_station', 'end_station', 'status']].to_string())
    
    # Analyze redundancy
    redundancy = network.analyze_network_redundancy()
    print(f"\nNetwork Redundancy Analysis:")
    print(f"Connected: {redundancy['connectivity']['is_connected']}")
    print(f"Critical nodes: {redundancy['critical_nodes']}")
    print(f"Critical edges: {redundancy['critical_edges']}")
    
    # Get performance metrics
    performance = network.get_network_performance_metrics()
    print(f"\nNetwork Performance:")
    print(f"Overall health: {performance['overall_health']}")
    print(f"Total capacity: {performance['aggregate_metrics']['total_capacity_gbps']} Gbps")
    print(f"Average utilization: {performance['aggregate_metrics']['average_utilization']:.1f}%")


if __name__ == "__main__":
    main()
