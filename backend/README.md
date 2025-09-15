# Backend - AI Network Stabilization API

This directory contains the backend API server for the AI Network Stabilization Dashboard.

## Structure

```
backend/
├── server.py              # Main Flask application
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── __init__.py           # Package initialization
└── README.md             # This file
```

## Features

- **RESTful API**: Clean API endpoints for frontend communication
- **Real-time Simulation**: Continuous network simulation with AI integration
- **AI Integration**: Anomaly detection and network analysis
- **CORS Support**: Cross-origin requests for frontend
- **Error Handling**: Robust error handling and logging
- **Configuration**: Centralized configuration management

## API Endpoints

### Core Endpoints
- `GET /` - Serve the main dashboard
- `GET /api/metrics` - Get current network metrics
- `GET /api/alerts` - Get system alerts

### Control Endpoints
- `POST /api/start` - Start simulation
- `POST /api/stop` - Stop simulation
- `POST /api/reset` - Reset simulation
- `POST /api/switch/<path>` - Switch network path

## Dependencies

- **Flask**: Web framework
- **Flask-CORS**: Cross-origin resource sharing
- **TensorFlow**: AI model inference
- **scikit-learn**: Data preprocessing
- **pandas**: Data manipulation
- **numpy**: Numerical computing

## Installation

```bash
cd backend
pip install -r requirements.txt
```

## Usage

### Development
```bash
python server.py
```

### Production
```bash
gunicorn -w 4 -b 0.0.0.0:5000 server:app
```

## Configuration

Edit `config.py` to modify:
- Server settings (host, port, debug)
- API configuration (CORS, timeouts)
- Simulation parameters
- AI model paths
- Logging settings

## AI Models

The backend integrates with pre-trained AI models:
- **Anomaly Detector**: LSTM Autoencoder for anomaly detection
- **Network Analyzer**: LSTM Classifier for path prediction
- **Fallback Heuristics**: Rule-based fallbacks when models fail

## Performance

- **Update Interval**: 3 seconds (configurable)
- **Data Points**: Limited to 15 for performance
- **Alerts**: Maximum 10 alerts in memory
- **Threading**: Multi-threaded for concurrent requests

## Error Handling

- **Model Loading**: Graceful fallback to heuristics
- **API Errors**: Proper HTTP status codes
- **Simulation Errors**: Automatic recovery
- **Logging**: Comprehensive error logging

## Security

- **CORS**: Configured for specific origins
- **Input Validation**: All inputs validated
- **Error Messages**: Sanitized error responses
- **Rate Limiting**: Built-in request throttling
