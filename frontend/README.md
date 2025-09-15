# Frontend - AI Network Stabilization Dashboard

This directory contains the frontend components for the AI Network Stabilization Dashboard.

## Structure

```
frontend/
├── index.html              # Main HTML file
├── assets/
│   ├── css/
│   │   └── dashboard.css   # Dashboard styles
│   ├── js/
│   │   └── dashboard.js    # Dashboard JavaScript
│   └── images/             # Static images (if any)
└── README.md              # This file
```

## Features

- **Responsive Design**: Works on desktop, tablet, and mobile
- **Real-time Updates**: Live network metrics and charts
- **3D Visualization**: Interactive globe showing satellite network topology
- **Interactive Charts**: Latency, packet loss, bandwidth, and cost visualizations
- **Alert System**: Real-time notifications for network events
- **Modern UI**: Glassmorphism design with dark theme

## Dependencies

- **Chart.js**: For interactive charts and graphs
- **Plotly.js**: For 3D globe visualization
- **Font Awesome**: For icons
- **Modern CSS**: Flexbox, Grid, and CSS animations

## Usage

1. **Development**: Open `index.html` in a web browser
2. **Production**: Serve through the backend API server
3. **Backend Integration**: Connects to Flask API at `http://localhost:5000`

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Customization

- **Styling**: Modify `assets/css/dashboard.css`
- **Functionality**: Update `assets/js/dashboard.js`
- **Layout**: Edit `index.html`

## Performance

- **Optimized**: Minimal external dependencies
- **Efficient**: 3-second update intervals
- **Responsive**: Smooth animations and transitions
- **Lightweight**: Fast loading and rendering
