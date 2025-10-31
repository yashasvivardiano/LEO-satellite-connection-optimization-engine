#!/usr/bin/env python3
"""
Fix AI Models Script

This script creates simple working models to replace the corrupted ones,
or provides better fallback functionality.
"""

import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler

def create_simple_autoencoder():
    """Create a simple working autoencoder model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(12, 10)),  # window_size, features
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16, return_sequences=True),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(10))
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def create_simple_classifier():
    """Create a simple working classifier model."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(12, 10)),  # window_size, features
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(2, activation='softmax')  # wired/satellite
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_dummy_data():
    """Create dummy training data."""
    np.random.seed(42)
    
    # Generate synthetic network data
    n_samples = 1000
    n_features = 10
    window_size = 12
    
    # Feature names from manifest
    feature_names = [
        'wired_latency_ms', 'satellite_latency_ms',
        'wired_jitter_ms', 'satellite_jitter_ms',
        'wired_packet_loss_pct', 'satellite_packet_loss_pct',
        'wired_bandwidth_mbps', 'satellite_bandwidth_mbps',
        'wired_quality_cost', 'satellite_quality_cost'
    ]
    
    # Generate realistic network data
    data = []
    for i in range(n_samples):
        # Wired metrics (generally better)
        wired_latency = np.random.normal(25, 5)
        wired_jitter = np.random.normal(2, 0.5)
        wired_loss = np.random.exponential(0.1)
        wired_bandwidth = np.random.normal(100, 10)
        wired_cost = np.random.normal(0.3, 0.1)
        
        # Satellite metrics (higher latency, more variable)
        satellite_latency = np.random.normal(45, 10)
        satellite_jitter = np.random.normal(5, 2)
        satellite_loss = np.random.exponential(0.2)
        satellite_bandwidth = np.random.normal(80, 15)
        satellite_cost = np.random.normal(0.5, 0.15)
        
        data.append([
            wired_latency, satellite_latency,
            wired_jitter, satellite_jitter,
            wired_loss, satellite_loss,
            wired_bandwidth, satellite_bandwidth,
            wired_cost, satellite_cost
        ])
    
    return np.array(data), feature_names

def main():
    """Main function to fix the models."""
    print("🔧 Fixing AI Models...")
    
    model_dir = Path("models/dual_ai")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dummy data
    print("📊 Generating training data...")
    data, feature_names = create_dummy_data()
    
    # Create scaler
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Create windowed data for LSTM
    window_size = 12
    X_windows = []
    y_labels = []
    
    for i in range(len(scaled_data) - window_size):
        X_windows.append(scaled_data[i:i+window_size])
        # Simple heuristic: choose wired if wired_cost < satellite_cost
        wired_cost = data[i+window_size-1][8]  # wired_quality_cost
        satellite_cost = data[i+window_size-1][9]  # satellite_quality_cost
        y_labels.append([1, 0] if wired_cost < satellite_cost else [0, 1])
    
    X_windows = np.array(X_windows)
    y_labels = np.array(y_labels)
    
    print(f"📈 Created {len(X_windows)} training samples")
    
    # Train autoencoder
    print("🤖 Training autoencoder...")
    autoencoder = create_simple_autoencoder()
    
    # Train on reconstruction task
    autoencoder.fit(X_windows, X_windows, epochs=5, batch_size=32, verbose=1)
    
    # Calculate reconstruction errors for threshold
    reconstructions = autoencoder.predict(X_windows)
    reconstruction_errors = np.mean(np.square(X_windows - reconstructions), axis=(1, 2))
    threshold = np.percentile(reconstruction_errors, 99)
    
    print(f"📊 Anomaly threshold: {threshold:.4f}")
    
    # Train classifier
    print("🎯 Training classifier...")
    classifier = create_simple_classifier()
    
    classifier.fit(X_windows, y_labels, epochs=5, batch_size=32, verbose=1)
    
    # Save models
    print("💾 Saving models...")
    
    # Save autoencoder
    autoencoder.save(str(model_dir / "autoencoder.keras"))
    
    # Save classifier
    classifier.save(str(model_dir / "predictive_clf.keras"))
    
    # Save scaler
    joblib.dump(scaler, str(model_dir / "scaler.pkl"))
    
    # Save metrics
    anomaly_metrics = {
        "train_99pct_threshold": float(threshold),
        "model_type": "simple_lstm_autoencoder",
        "training_samples": len(X_windows)
    }
    
    with open(model_dir / "anomaly_metrics.json", 'w') as f:
        json.dump(anomaly_metrics, f, indent=2)
    
    # Save manifest
    manifest = {
        "tensorflow": "2.15.0",
        "pandas": "2.2.3",
        "numpy": "1.26.4",
        "feature_columns": feature_names,
        "window": window_size,
        "horizon": 6,
        "model_type": "simple_models",
        "training_date": "2024-10-29"
    }
    
    with open(model_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Test loading
    print("🧪 Testing model loading...")
    try:
        test_autoencoder = tf.keras.models.load_model(str(model_dir / "autoencoder.keras"))
        test_classifier = tf.keras.models.load_model(str(model_dir / "predictive_clf.keras"))
        test_scaler = joblib.load(str(model_dir / "scaler.pkl"))
        
        print("✅ All models loaded successfully!")
        print(f"   Autoencoder input shape: {test_autoencoder.input_shape}")
        print(f"   Classifier input shape: {test_classifier.input_shape}")
        print(f"   Scaler features: {test_scaler.n_features_in_}")
        
    except Exception as e:
        print(f"❌ Model loading test failed: {e}")
    
    print("🎉 Model fixing complete!")

if __name__ == "__main__":
    main()
