# 🛰️ AI Network Stabilization Tool - MVP

## 🎯 Project Overview

This MVP demonstrates the core concepts of an AI-driven network stabilization tool that transforms network management from reactive to proactive, self-optimizing paradigms. The system manages hybrid networks combining traditional wired infrastructure with LEO satellite communication links.

## ✨ Key Features

### 🔄 **Dual-AI Strategy**
- **Proactive Management**: Predicts routine network events and automatically switches paths
- **Reactive Containment**: Instantly detects and contains major, unforeseen failures

### 🌐 **Hybrid Network Simulation**
- Real-time simulation of wired and satellite communication paths
- Realistic telemetry generation with configurable parameters
- Event simulation including satellite handoffs, congestion, and failures

### 🤖 **AI-Powered Decision Making**
- **Predictive Steering Model**: LSTM classifier for optimal path prediction
- **Anomaly Detection Model**: LSTM Autoencoder for failure detection
- **Critical Failure Detector**: Multi-criteria failure identification

### 📊 **Real-Time Dashboard**
- Live network performance visualization
- AI decision tracking and confidence metrics
- Alert system for anomalies and critical failures
- Interactive controls for manual override

## 🚀 Quick Start

### 1. Test Core Components
```bash
python test_mvp_core.py
```

### 2. Launch MVP Dashboard
```bash
python run_mvp_dashboard.py
```

The dashboard will open at: http://localhost:8501

## 📁 Project Structure

```
LEO/
├── src/
│   ├── simulation/
│   │   └── satellite_link.py          # Hybrid network simulator
│   ├── ai/
│   │   ├── anomaly_detector.py        # AI anomaly detection
│   │   └── network_analyzer.py        # AI path prediction
│   └── monitoring/
│       └── dashboard.py               # Streamlit MVP dashboard
├── models/dual_ai/                    # Trained AI models
├── test_mvp_core.py                   # Core component tests
├── run_mvp_dashboard.py               # Dashboard launcher
└── MVP_README.md                      # This file
```

## 🔧 Core Components

### 1. **Hybrid Network Simulator** (`src/simulation/satellite_link.py`)
- Generates realistic network telemetry for wired and satellite paths
- Simulates predictable events (handoffs, congestion) and unpredictable failures
- Provides path switching and optimal path determination

### 2. **AI Anomaly Detector** (`src/ai/anomaly_detector.py`)
- Integrates trained LSTM Autoencoder for anomaly detection
- Real-time reconstruction error calculation
- Critical failure detection with multiple criteria

### 3. **Predictive Network Analyzer** (`src/ai/network_analyzer.py`)
- Integrates trained LSTM Classifier for path prediction
- Intelligent path switching recommendations
- Network health scoring and insights

### 4. **MVP Dashboard** (`src/monitoring/dashboard.py`)
- Real-time network performance visualization
- AI decision tracking and confidence metrics
- Interactive controls and alert system

## 🎮 Dashboard Features

### **Real-Time Monitoring**
- Live network metrics (latency, jitter, packet loss, bandwidth)
- Path switching visualization
- AI confidence indicators

### **Interactive Controls**
- Start/Stop simulation
- Manual path switching
- Auto-switching toggle
- Reset simulation

### **Alert System**
- Proactive switch notifications
- Anomaly detection alerts
- Critical failure warnings

## 📊 MVP Demonstration

The MVP demonstrates the core "Analyze → Predict → Stabilize" loop:

1. **Analyze**: Continuous monitoring of network performance
2. **Predict**: AI models predict optimal paths and detect anomalies
3. **Stabilize**: Automated path switching and failure containment

### **Predictable Events (Proactive)**
- Satellite handoff simulation with temporary degradation
- Wired congestion during peak hours
- AI prediction of optimal paths 60 seconds ahead
- Automated path switching based on AI recommendations

### **Unpredictable Failures (Reactive)**
- Random network failure injection
- LSTM Autoencoder anomaly detection
- Critical failure detection with multiple criteria
- Real-time alert generation

## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_mvp_core.py
```

Tests include:
- Hybrid Network Simulator functionality
- AI Anomaly Detector integration
- Predictive Network Analyzer capabilities
- Integrated system testing

## 🔮 Next Steps

This MVP successfully demonstrates the core concepts and is ready for:

1. **Phase 2**: Advanced modeling with Transformers and Reinforcement Learning
2. **Phase 3**: Real-world integration with live network hardware
3. **Phase 4**: Full-scale stabilization and commercialization

## 📈 Performance Metrics

The MVP tracks:
- Network performance metrics (latency, jitter, loss, bandwidth)
- AI prediction accuracy and confidence
- Path switching frequency and success rate
- Anomaly detection rate and false positives
- Critical failure detection and response time

## 🛠️ Technical Stack

- **Python 3.8+**
- **TensorFlow/Keras** for AI models
- **Streamlit** for dashboard
- **Plotly** for visualization
- **NumPy/Pandas** for data processing
- **SciPy** for signal processing

## 📝 Notes

- The AI models may have compatibility issues with newer TensorFlow versions
- The system gracefully handles model loading failures with fallback behavior
- All components are designed for easy integration and extension

## 🎉 Success Criteria Met

✅ **Hybrid Network Simulation**: Real-time simulation of wired + satellite paths  
✅ **Dual-AI Strategy**: Both predictive and anomaly detection models integrated  
✅ **Proactive Management**: Automated path switching based on AI predictions  
✅ **Reactive Containment**: Real-time anomaly and critical failure detection  
✅ **MVP Dashboard**: Interactive demonstration of all capabilities  
✅ **End-to-End Testing**: Comprehensive test suite validates all components  

The MVP successfully demonstrates the core vision of transforming network management from reactive to proactive, self-optimizing paradigms using AI-driven decision making.
