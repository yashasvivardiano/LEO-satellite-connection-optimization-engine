#!/usr/bin/env python3
"""
Hybrid Network Simulation Engine

This module provides real-time simulation of hybrid networks combining
wired and satellite communication paths. It generates realistic network
telemetry data and simulates both predictable events and unpredictable failures.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network performance metrics for a single path."""
    latency_ms: float
    jitter_ms: float
    packet_loss_pct: float
    bandwidth_mbps: float
    quality_cost: float
    timestamp: datetime

@dataclass
class NetworkEvent:
    """Represents a network event (predictable or unpredictable)."""
    event_type: str  # 'handoff', 'congestion', 'failure', 'recovery'
    path: str  # 'wired' or 'satellite'
    severity: float  # 0.0 to 1.0
    duration_seconds: int
    start_time: datetime
    description: str

class HybridNetworkSimulator:
    """
    Simulates a hybrid network with wired and satellite paths.
    Generates realistic telemetry data and network events.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the simulator with configuration."""
        self.config = self._load_config(config_path)
        self.current_time = datetime.now()
        self.simulation_start = self.current_time
        
        # Network state
        self.active_path = "wired"  # Current active path
        self.path_metrics = {
            "wired": self._initialize_wired_metrics(),
            "satellite": self._initialize_satellite_metrics()
        }
        
        # Event tracking
        self.scheduled_events = []
        self.active_events = []
        self.event_history = []
        
        # Simulation parameters
        self.time_step = 10  # seconds between updates
        self.simulation_speed = 1.0  # real-time multiplier
        
        logger.info("Hybrid Network Simulator initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load simulation configuration."""
        default_config = {
            "wired": {
                "base_latency_ms": 30.0,
                "base_jitter_ms": 6.0,
                "base_packet_loss_pct": 0.015,
                "base_bandwidth_mbps": 100.0,
                "congestion_hours": [9, 10, 11, 14, 15, 16, 17],  # Peak hours
                "congestion_multiplier": 1.5
            },
            "satellite": {
                "base_latency_ms": 45.0,
                "base_jitter_ms": 8.0,
                "base_packet_loss_pct": 0.05,
                "base_bandwidth_mbps": 80.0,
                "handoff_interval_minutes": 15,  # Average time between handoffs
                "handoff_duration_seconds": 30,
                "elevation_deg": 45.0,
                "speed_km_s": 7.8
            },
            "anomaly": {
                "failure_probability": 0.001,  # 0.1% chance per time step
                "failure_duration_minutes": 5,
                "failure_severity_range": [0.7, 1.0]
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_wired_metrics(self) -> NetworkMetrics:
        """Initialize wired path with baseline metrics."""
        config = self.config["wired"]
        return NetworkMetrics(
            latency_ms=config["base_latency_ms"],
            jitter_ms=config["base_jitter_ms"],
            packet_loss_pct=config["base_packet_loss_pct"],
            bandwidth_mbps=config["base_bandwidth_mbps"],
            quality_cost=self._calculate_quality_cost("wired", config),
            timestamp=self.current_time
        )
    
    def _initialize_satellite_metrics(self) -> NetworkMetrics:
        """Initialize satellite path with baseline metrics."""
        config = self.config["satellite"]
        return NetworkMetrics(
            latency_ms=config["base_latency_ms"],
            jitter_ms=config["base_jitter_ms"],
            packet_loss_pct=config["base_packet_loss_pct"],
            bandwidth_mbps=config["base_bandwidth_mbps"],
            quality_cost=self._calculate_quality_cost("satellite", config),
            timestamp=self.current_time
        )
    
    def _calculate_quality_cost(self, path: str, config: Dict) -> float:
        """Calculate quality cost based on network metrics."""
        # Simple quality cost calculation
        latency_cost = config["base_latency_ms"] / 100.0
        jitter_cost = config["base_jitter_ms"] / 10.0
        loss_cost = config["base_packet_loss_pct"] * 10.0
        bandwidth_cost = (100.0 - config["base_bandwidth_mbps"]) / 100.0
        
        return latency_cost + jitter_cost + loss_cost + bandwidth_cost
    
    def generate_telemetry(self) -> Dict[str, NetworkMetrics]:
        """Generate current network telemetry for both paths."""
        self.current_time += timedelta(seconds=self.time_step)
        
        # Update wired path metrics
        self.path_metrics["wired"] = self._update_path_metrics("wired")
        
        # Update satellite path metrics
        self.path_metrics["satellite"] = self._update_path_metrics("satellite")
        
        # Check for and apply events
        self._process_events()
        
        # Schedule new events
        self._schedule_events()
        
        return self.path_metrics.copy()
    
    def _update_path_metrics(self, path: str) -> NetworkMetrics:
        """Update metrics for a specific path."""
        current_metrics = self.path_metrics[path]
        config = self.config[path]
        
        # Base values with some random variation
        base_latency = config["base_latency_ms"]
        base_jitter = config["base_jitter_ms"]
        base_loss = config["base_packet_loss_pct"]
        base_bandwidth = config["base_bandwidth_mbps"]
        
        # Add random variation (5% standard deviation)
        latency = max(1.0, np.random.normal(base_latency, base_latency * 0.05))
        jitter = max(0.1, np.random.normal(base_jitter, base_jitter * 0.05))
        loss = max(0.0, np.random.normal(base_loss, base_loss * 0.05))
        bandwidth = max(1.0, np.random.normal(base_bandwidth, base_bandwidth * 0.05))
        
        # Apply time-based effects
        if path == "wired":
            latency, jitter, loss, bandwidth = self._apply_wired_effects(
                latency, jitter, loss, bandwidth
            )
        elif path == "satellite":
            latency, jitter, loss, bandwidth = self._apply_satellite_effects(
                latency, jitter, loss, bandwidth
            )
        
        # Apply active events
        for event in self.active_events:
            if event.path == path:
                latency, jitter, loss, bandwidth = self._apply_event_effects(
                    event, latency, jitter, loss, bandwidth
                )
        
        # Calculate quality cost
        quality_cost = self._calculate_quality_cost(path, {
            "base_latency_ms": latency,
            "base_jitter_ms": jitter,
            "base_packet_loss_pct": loss,
            "base_bandwidth_mbps": bandwidth
        })
        
        return NetworkMetrics(
            latency_ms=latency,
            jitter_ms=jitter,
            packet_loss_pct=loss,
            bandwidth_mbps=bandwidth,
            quality_cost=quality_cost,
            timestamp=self.current_time
        )
    
    def _apply_wired_effects(self, latency: float, jitter: float, 
                           loss: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """Apply time-based effects to wired path."""
        config = self.config["wired"]
        current_hour = self.current_time.hour
        
        # Peak hour congestion
        if current_hour in config["congestion_hours"]:
            congestion_factor = config["congestion_multiplier"]
            latency *= congestion_factor
            jitter *= congestion_factor
            loss *= congestion_factor
            bandwidth /= congestion_factor
        
        return latency, jitter, loss, bandwidth
    
    def _apply_satellite_effects(self, latency: float, jitter: float,
                               loss: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """Apply satellite-specific effects."""
        config = self.config["satellite"]
        
        # Simulate elevation-based effects
        elevation = config["elevation_deg"]
        if elevation < 30:
            # Low elevation - worse performance
            latency *= 1.2
            jitter *= 1.5
            loss *= 2.0
        elif elevation > 60:
            # High elevation - better performance
            latency *= 0.9
            jitter *= 0.8
            loss *= 0.7
        
        # Simulate orbital speed effects
        speed = config["speed_km_s"]
        if speed > 8.0:
            # High speed - more jitter
            jitter *= 1.3
        
        return latency, jitter, loss, bandwidth
    
    def _apply_event_effects(self, event: NetworkEvent, latency: float, jitter: float,
                           loss: float, bandwidth: float) -> Tuple[float, float, float, float]:
        """Apply effects from active events."""
        severity = event.severity
        
        if event.event_type == "handoff":
            # Satellite handoff - temporary degradation
            latency *= (1 + severity * 0.5)
            jitter *= (1 + severity * 2.0)
            loss *= (1 + severity * 3.0)
            
        elif event.event_type == "congestion":
            # Network congestion
            latency *= (1 + severity * 0.8)
            jitter *= (1 + severity * 1.2)
            loss *= (1 + severity * 2.0)
            bandwidth *= (1 - severity * 0.3)
            
        elif event.event_type == "failure":
            # Network failure - severe degradation
            latency *= (1 + severity * 5.0)
            jitter *= (1 + severity * 10.0)
            loss = min(100.0, loss * (1 + severity * 20.0))
            bandwidth *= (1 - severity * 0.8)
        
        return latency, jitter, loss, bandwidth
    
    def _process_events(self):
        """Process active events and remove expired ones."""
        current_events = []
        
        for event in self.active_events:
            # Check if event has expired
            if self.current_time >= event.start_time + timedelta(seconds=event.duration_seconds):
                logger.info(f"Event expired: {event.description}")
                self.event_history.append(event)
            else:
                current_events.append(event)
        
        self.active_events = current_events
    
    def _schedule_events(self):
        """Schedule new events based on probability and patterns."""
        # Schedule satellite handoffs
        if np.random.random() < 0.1:  # 10% chance per time step
            self._schedule_handoff_event()
        
        # Schedule wired congestion
        if np.random.random() < 0.05:  # 5% chance per time step
            self._schedule_congestion_event()
        
        # Schedule random failures
        failure_prob = self.config["anomaly"]["failure_probability"]
        if np.random.random() < failure_prob:
            self._schedule_failure_event()
    
    def _schedule_handoff_event(self):
        """Schedule a satellite handoff event."""
        config = self.config["satellite"]
        duration = config["handoff_duration_seconds"]
        severity = np.random.uniform(0.3, 0.7)
        
        event = NetworkEvent(
            event_type="handoff",
            path="satellite",
            severity=severity,
            duration_seconds=duration,
            start_time=self.current_time,
            description=f"Satellite handoff - {duration}s duration"
        )
        
        self.active_events.append(event)
        logger.info(f"Scheduled handoff event: {event.description}")
    
    def _schedule_congestion_event(self):
        """Schedule a wired congestion event."""
        config = self.config["wired"]
        duration = np.random.randint(60, 300)  # 1-5 minutes
        severity = np.random.uniform(0.2, 0.6)
        
        event = NetworkEvent(
            event_type="congestion",
            path="wired",
            severity=severity,
            duration_seconds=duration,
            start_time=self.current_time,
            description=f"Wired congestion - {duration}s duration"
        )
        
        self.active_events.append(event)
        logger.info(f"Scheduled congestion event: {event.description}")
    
    def _schedule_failure_event(self):
        """Schedule a random network failure event."""
        config = self.config["anomaly"]
        duration = config["failure_duration_minutes"] * 60
        severity_range = config["failure_severity_range"]
        severity = np.random.uniform(severity_range[0], severity_range[1])
        
        # Randomly choose which path fails
        path = np.random.choice(["wired", "satellite"])
        
        event = NetworkEvent(
            event_type="failure",
            path=path,
            severity=severity,
            duration_seconds=duration,
            start_time=self.current_time,
            description=f"CRITICAL: {path} path failure - {duration}s duration"
        )
        
        self.active_events.append(event)
        logger.warning(f"Scheduled failure event: {event.description}")
    
    def get_optimal_path(self) -> str:
        """Determine the optimal path based on current metrics."""
        wired_cost = self.path_metrics["wired"].quality_cost
        satellite_cost = self.path_metrics["satellite"].quality_cost
        
        # Choose path with lower quality cost
        if wired_cost < satellite_cost:
            return "wired"
        else:
            return "satellite"
    
    def switch_path(self, new_path: str) -> bool:
        """Switch to a new active path."""
        if new_path in self.path_metrics:
            old_path = self.active_path
            self.active_path = new_path
            logger.info(f"Path switched: {old_path} -> {new_path}")
            return True
        return False
    
    def get_simulation_data(self) -> Dict:
        """Get current simulation state for AI models."""
        wired = self.path_metrics["wired"]
        satellite = self.path_metrics["satellite"]
        
        return {
            "wired_latency_ms": wired.latency_ms,
            "wired_jitter_ms": wired.jitter_ms,
            "wired_packet_loss_pct": wired.packet_loss_pct,
            "wired_bandwidth_mbps": wired.bandwidth_mbps,
            "wired_quality_cost": wired.quality_cost,
            "satellite_latency_ms": satellite.latency_ms,
            "satellite_jitter_ms": satellite.jitter_ms,
            "satellite_packet_loss_pct": satellite.packet_loss_pct,
            "satellite_bandwidth_mbps": satellite.bandwidth_mbps,
            "satellite_quality_cost": satellite.quality_cost,
            "current_optimal_path": self.get_optimal_path(),
            "active_path": self.active_path,
            "timestamp": self.current_time,
            "active_events": len(self.active_events),
            "simulation_time": (self.current_time - self.simulation_start).total_seconds()
        }
    
    def get_event_summary(self) -> Dict:
        """Get summary of current events and network state."""
        return {
            "active_events": [
                {
                    "type": event.event_type,
                    "path": event.path,
                    "severity": event.severity,
                    "description": event.description,
                    "remaining_seconds": event.duration_seconds - 
                        (self.current_time - event.start_time).total_seconds()
                }
                for event in self.active_events
            ],
            "event_history_count": len(self.event_history),
            "active_path": self.active_path,
            "optimal_path": self.get_optimal_path(),
            "path_match": self.active_path == self.get_optimal_path()
        }

def create_simulator(config_path: Optional[str] = None) -> HybridNetworkSimulator:
    """Create a new hybrid network simulator instance."""
    return HybridNetworkSimulator(config_path)

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    simulator = create_simulator()
    
    print("🛰️ Hybrid Network Simulator Demo")
    print("=" * 50)
    
    for i in range(10):
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