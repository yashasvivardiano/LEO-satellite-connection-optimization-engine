#!/usr/bin/env python3
"""
AI Network Analysis Module

This module integrates the trained predictive steering model for
intelligent path selection and network optimization.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import joblib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PredictiveNetworkAnalyzer:
    """
    Real-time network analysis using trained LSTM classifier.
    Predicts optimal network paths and provides intelligent steering.
    """
    
    def __init__(self, model_dir: str = "models/dual_ai"):
        """Initialize the network analyzer with trained models."""
        self.model_dir = Path(model_dir)
        self.predictive_model = None
        self.scaler = None
        self.feature_columns = None
        self.window_size = 12
        self.horizon = 6
        self.is_initialized = False
        
        # Load models and configuration
        self._load_models()
        
        # Analysis state
        self.recent_windows = []
        self.prediction_history = []
        self.switch_count = 0
        
        logger.info("Predictive Network Analyzer initialized")
    
    def _load_models(self):
        """Load the trained predictive model and scaler."""
        try:
            # Load predictive model
            model_path = self.model_dir / "predictive_clf.keras"
            if model_path.exists():
                self.predictive_model = tf.keras.models.load_model(str(model_path))
                logger.info(f"Loaded predictive model from {model_path}")
            else:
                logger.error(f"Predictive model not found at {model_path}")
                return
            
            # Load scaler
            scaler_path = self.model_dir / "scaler.pkl"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info(f"Loaded scaler from {scaler_path}")
            else:
                logger.error(f"Scaler not found at {scaler_path}")
                return
            
            # Load feature configuration
            manifest_path = self.model_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                self.feature_columns = manifest.get('feature_columns', [])
                self.window_size = manifest.get('window', 12)
                self.horizon = manifest.get('horizon', 6)
                logger.info(f"Loaded feature config: {len(self.feature_columns)} features")
            else:
                logger.warning("Manifest not found, using default features")
                self.feature_columns = [
                    'wired_latency_ms', 'satellite_latency_ms',
                    'wired_jitter_ms', 'satellite_jitter_ms',
                    'wired_packet_loss_pct', 'satellite_packet_loss_pct',
                    'wired_bandwidth_mbps', 'satellite_bandwidth_mbps',
                    'wired_quality_cost', 'satellite_quality_cost'
                ]
            
            self.is_initialized = True
            logger.info("Predictive network analyzer successfully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize predictive analyzer: {e}")
            self.is_initialized = False
    
    def predict_optimal_path(self, network_data: Dict) -> Dict:
        """
        Predict the optimal network path for the next time horizon.
        
        Args:
            network_data: Dictionary containing current network metrics
            
        Returns:
            Dictionary with path prediction results
        """
        if not self.is_initialized:
            # Fallback: Use simple heuristic based on quality costs
            wired_cost = network_data.get('wired_quality_cost', 1.0)
            satellite_cost = network_data.get('satellite_quality_cost', 1.0)
            
            if wired_cost < satellite_cost:
                predicted_path = "wired"
                confidence = min(0.9, max(0.6, 1.0 - abs(wired_cost - satellite_cost)))
            else:
                predicted_path = "satellite"
                confidence = min(0.9, max(0.6, 1.0 - abs(wired_cost - satellite_cost)))
            
            return {
                "predicted_path": predicted_path,
                "confidence": confidence,
                "probabilities": {"wired": 0.5, "satellite": 0.5},
                "status": "Fallback heuristic"
            }
        
        try:
            # Extract features from network data
            features = self._extract_features(network_data)
            
            # Add to recent windows
            self.recent_windows.append(features)
            
            # Keep only recent windows
            if len(self.recent_windows) > self.window_size:
                self.recent_windows = self.recent_windows[-self.window_size:]
            
            # Need enough data for prediction
            if len(self.recent_windows) < self.window_size:
                return {
                    "predicted_path": "wired",
                    "confidence": 0.0,
                    "probabilities": {"wired": 0.5, "satellite": 0.5},
                    "status": "Insufficient data"
                }
            
            # Prepare data for model
            window_data = np.array(self.recent_windows[-self.window_size:])
            window_data = window_data.reshape(1, self.window_size, -1)
            
            # Get prediction probabilities
            probabilities = self.predictive_model.predict(window_data, verbose=0)[0]
            
            # Convert to path names
            path_names = ["wired", "satellite"]
            path_probs = {
                path_names[i]: float(probabilities[i]) 
                for i in range(len(path_names))
            }
            
            # Get predicted path (highest probability)
            predicted_idx = np.argmax(probabilities)
            predicted_path = path_names[predicted_idx]
            confidence = float(probabilities[predicted_idx])
            
            # Log prediction
            prediction_info = {
                "timestamp": datetime.now(),
                "predicted_path": predicted_path,
                "confidence": confidence,
                "probabilities": path_probs,
                "network_data": network_data
            }
            self.prediction_history.append(prediction_info)
            
            logger.info(f"Path prediction: {predicted_path} (confidence: {confidence:.3f})")
            
            return {
                "predicted_path": predicted_path,
                "confidence": confidence,
                "probabilities": path_probs,
                "status": "Success"
            }
            
        except Exception as e:
            logger.error(f"Error in path prediction: {e}")
            return {
                "predicted_path": "wired",
                "confidence": 0.0,
                "probabilities": {"wired": 0.5, "satellite": 0.5},
                "error": str(e)
            }
    
    def _extract_features(self, network_data: Dict) -> List[float]:
        """Extract feature vector from network data."""
        features = []
        
        for col in self.feature_columns:
            if col in network_data:
                features.append(float(network_data[col]))
            else:
                logger.warning(f"Feature {col} not found in network data")
                features.append(0.0)
        
        return features
    
    def should_switch_path(self, current_path: str, network_data: Dict, 
                          confidence_threshold: float = 0.7) -> Dict:
        """
        Determine if the network should switch paths based on prediction.
        
        Args:
            current_path: Current active path
            network_data: Current network metrics
            confidence_threshold: Minimum confidence for switching
            
        Returns:
            Dictionary with switching recommendation
        """
        prediction = self.predict_optimal_path(network_data)
        
        if prediction.get("status") != "Success":
            return {
                "should_switch": False,
                "reason": "Prediction failed",
                "predicted_path": current_path,
                "confidence": 0.0
            }
        
        predicted_path = prediction["predicted_path"]
        confidence = prediction["confidence"]
        
        # Check if switch is recommended
        should_switch = (predicted_path != current_path and 
                        confidence >= confidence_threshold)
        
        reason = "No switch needed"
        if predicted_path == current_path:
            reason = "Predicted path matches current path"
        elif confidence < confidence_threshold:
            reason = f"Confidence too low: {confidence:.3f} < {confidence_threshold}"
        else:
            reason = f"Switch recommended: {current_path} -> {predicted_path}"
            self.switch_count += 1
        
        return {
            "should_switch": should_switch,
            "reason": reason,
            "predicted_path": predicted_path,
            "confidence": confidence,
            "current_path": current_path
        }
    
    def get_network_insights(self, network_data: Dict) -> Dict:
        """
        Get comprehensive network insights and recommendations.
        
        Args:
            network_data: Current network metrics
            
        Returns:
            Dictionary with network insights
        """
        prediction = self.predict_optimal_path(network_data)
        
        # Analyze current network state
        wired_cost = network_data.get("wired_quality_cost", 0)
        satellite_cost = network_data.get("satellite_quality_cost", 0)
        
        # Calculate quality difference
        cost_difference = abs(wired_cost - satellite_cost)
        better_path = "wired" if wired_cost < satellite_cost else "satellite"
        
        # Determine network health
        health_score = self._calculate_health_score(network_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(network_data, prediction)
        
        return {
            "prediction": prediction,
            "current_analysis": {
                "wired_cost": wired_cost,
                "satellite_cost": satellite_cost,
                "cost_difference": cost_difference,
                "better_path": better_path,
                "health_score": health_score
            },
            "recommendations": recommendations,
            "timestamp": datetime.now()
        }
    
    def _calculate_health_score(self, network_data: Dict) -> float:
        """Calculate overall network health score (0-100)."""
        try:
            # Extract key metrics
            wired_latency = network_data.get("wired_latency_ms", 0)
            satellite_latency = network_data.get("satellite_latency_ms", 0)
            wired_loss = network_data.get("wired_packet_loss_pct", 0)
            satellite_loss = network_data.get("satellite_packet_loss_pct", 0)
            
            # Calculate health components
            latency_score = 100 - min(100, (wired_latency + satellite_latency) / 2)
            loss_score = 100 - min(100, (wired_loss + satellite_loss) * 10)
            
            # Overall health score
            health_score = (latency_score + loss_score) / 2
            return max(0, min(100, health_score))
            
        except Exception as e:
            logger.error(f"Error calculating health score: {e}")
            return 50.0  # Default neutral score
    
    def _generate_recommendations(self, network_data: Dict, prediction: Dict) -> List[str]:
        """Generate network optimization recommendations."""
        recommendations = []
        
        # Check for high latency
        if network_data.get("wired_latency_ms", 0) > 50:
            recommendations.append("Consider switching to satellite due to high wired latency")
        
        if network_data.get("satellite_latency_ms", 0) > 100:
            recommendations.append("Consider switching to wired due to high satellite latency")
        
        # Check for packet loss
        if network_data.get("wired_packet_loss_pct", 0) > 1.0:
            recommendations.append("High packet loss on wired path - monitor closely")
        
        if network_data.get("satellite_packet_loss_pct", 0) > 2.0:
            recommendations.append("High packet loss on satellite path - consider switching")
        
        # Check prediction confidence
        if prediction.get("confidence", 0) < 0.6:
            recommendations.append("Low prediction confidence - gather more data")
        
        # Check for path mismatch
        if prediction.get("predicted_path") != network_data.get("active_path", "wired"):
            recommendations.append(f"AI recommends switching to {prediction.get('predicted_path')}")
        
        return recommendations
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of network analysis history."""
        recent_predictions = [p for p in self.prediction_history 
                            if (datetime.now() - p["timestamp"]).total_seconds() < 3600]
        
        return {
            "total_predictions": len(self.prediction_history),
            "recent_predictions": len(recent_predictions),
            "total_switches": self.switch_count,
            "is_initialized": self.is_initialized,
            "window_size": self.window_size,
            "horizon": self.horizon
        }

def create_network_analyzer(model_dir: str = "models/dual_ai") -> PredictiveNetworkAnalyzer:
    """Create a new network analyzer instance."""
    return PredictiveNetworkAnalyzer(model_dir)