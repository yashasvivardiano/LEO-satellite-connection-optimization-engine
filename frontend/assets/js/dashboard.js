// AI Network Stabilization Dashboard JavaScript

// Global variables
let simulationRunning = false;
let simulationInterval = null;
let chartData = {
    timestamps: [],
    wiredLatency: [],
    satelliteLatency: [],
    wiredLoss: [],
    satelliteLoss: [],
    wiredBandwidth: [],
    satelliteBandwidth: [],
    wiredCost: [],
    satelliteCost: []
};

// Global chart instances
let charts = {};

// Initialize charts
function initializeCharts() {
    try {
        // Latency Chart
        const latencyCtx = document.getElementById('latencyChart').getContext('2d');
        charts.latency = new Chart(latencyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Wired Latency',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    fill: false
                }, {
                    label: 'Satellite Latency',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false, // Disable animations for better performance
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });

        // Loss Chart
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        charts.loss = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Wired Loss',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    fill: false
                }, {
                    label: 'Satellite Loss',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });

        // Bandwidth Chart
        const bandwidthCtx = document.getElementById('bandwidthChart').getContext('2d');
        charts.bandwidth = new Chart(bandwidthCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Wired Bandwidth',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    fill: false
                }, {
                    label: 'Satellite Bandwidth',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });

        // Cost Chart
        const costCtx = document.getElementById('costChart').getContext('2d');
        charts.cost = new Chart(costCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Wired Cost',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    fill: false
                }, {
                    label: 'Satellite Cost',
                    data: [],
                    borderColor: '#ff6b6b',
                    backgroundColor: 'rgba(255, 107, 107, 0.1)',
                    tension: 0.4,
                    fill: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: false,
                plugins: {
                    legend: {
                        labels: { color: '#ffffff' }
                    }
                },
                scales: {
                    x: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    },
                    y: { 
                        ticks: { color: '#a0a0a0' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' }
                    }
                }
            }
        });

        console.log('Charts initialized successfully');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
}

// Initialize 3D Globe
function initializeGlobe() {
    try {
        // Hide loading spinner
        const loadingElement = document.querySelector('#globeChart .loading');
        if (loadingElement) {
            loadingElement.style.display = 'none';
        }

        const data = [{
            type: 'scatter3d',
            x: [0, 1, -1, 0.5, -0.5],
            y: [0, 0, 0, 0.8, -0.8],
            z: [0, 0, 0, 0.6, -0.6],
            mode: 'markers+text',
            marker: {
                size: [20, 15, 15, 12, 12],
                color: ['#00d4ff', '#ff6b6b', '#ff6b6b', '#ffa502', '#ffa502'],
                symbol: ['circle', 'diamond', 'diamond', 'diamond', 'diamond']
            },
            text: ['Earth', 'Sat-1', 'Sat-2', 'Sat-3', 'Sat-4'],
            textposition: 'top center'
        }];

        const layout = {
            title: 'Global Network Topology',
            scene: {
                xaxis: { showticklabels: false, showgrid: false, zeroline: false },
                yaxis: { showticklabels: false, showgrid: false, zeroline: false },
                zaxis: { showticklabels: false, showgrid: false, zeroline: false },
                bgcolor: 'rgba(0,0,0,0)',
                camera: { eye: { x: 1.5, y: 1.5, z: 1.5 } }
            },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' },
            height: 400
        };

        Plotly.newPlot('globeChart', data, layout, { responsive: true });
        console.log('3D Globe initialized successfully');
    } catch (error) {
        console.error('Error initializing 3D globe:', error);
        // Show fallback message
        document.getElementById('globeChart').innerHTML = '<div style="text-align: center; padding: 50px; color: #a0a0a0;">3D Globe temporarily unavailable</div>';
    }
}

// Fetch data from Python backend
async function fetchNetworkData() {
    try {
        const response = await fetch('/api/metrics');
        const data = await response.json();
        return {
            timestamp: new Date().toLocaleTimeString(),
            wiredLatency: data.wired_latency.toString(),
            satelliteLatency: data.satellite_latency.toString(),
            wiredLoss: data.wired_loss.toString(),
            satelliteLoss: data.satellite_loss.toString(),
            wiredBandwidth: data.wired_bandwidth.toString(),
            satelliteBandwidth: data.satellite_bandwidth.toString(),
            wiredCost: data.wired_cost.toString(),
            satelliteCost: data.satellite_cost.toString(),
            optimalPath: data.active_path,
            aiConfidence: data.ai_confidence.toString()
        };
    } catch (error) {
        console.error('Error fetching data:', error);
        return generateLocalData(); // Fallback
    }
}

// Generate local data for testing
function generateLocalData() {
    const baseTime = new Date();
    return {
        timestamp: baseTime.toLocaleTimeString(),
        wiredLatency: (30 + Math.random() * 10).toFixed(1),
        satelliteLatency: (40 + Math.random() * 15).toFixed(1),
        wiredLoss: (0.01 + Math.random() * 0.02).toFixed(3),
        satelliteLoss: (0.03 + Math.random() * 0.04).toFixed(3),
        wiredBandwidth: (950 + Math.random() * 50).toFixed(0),
        satelliteBandwidth: (800 + Math.random() * 100).toFixed(0),
        wiredCost: (0.8 + Math.random() * 0.3).toFixed(2),
        satelliteCost: (1.2 + Math.random() * 0.4).toFixed(2),
        optimalPath: Math.random() > 0.5 ? 'WIRED' : 'SATELLITE',
        aiConfidence: (60 + Math.random() * 30).toFixed(1)
    };
}

// Update metrics display
function updateMetrics(data) {
    document.getElementById('activePath').textContent = data.optimalPath;
    document.getElementById('wiredLatency').textContent = data.wiredLatency + ' ms';
    document.getElementById('satelliteLatency').textContent = data.satelliteLatency + ' ms';
    document.getElementById('aiConfidence').textContent = data.aiConfidence + '%';
}

// Update charts
function updateCharts(data) {
    try {
        // Add new data point
        chartData.timestamps.push(data.timestamp);
        chartData.wiredLatency.push(parseFloat(data.wiredLatency));
        chartData.satelliteLatency.push(parseFloat(data.satelliteLatency));
        chartData.wiredLoss.push(parseFloat(data.wiredLoss));
        chartData.satelliteLoss.push(parseFloat(data.satelliteLoss));
        chartData.wiredBandwidth.push(parseFloat(data.wiredBandwidth));
        chartData.satelliteBandwidth.push(parseFloat(data.satelliteBandwidth));
        chartData.wiredCost.push(parseFloat(data.wiredCost));
        chartData.satelliteCost.push(parseFloat(data.satelliteCost));
        
        // Keep only last 15 data points (reduced for better performance)
        const maxPoints = 15;
        if (chartData.timestamps.length > maxPoints) {
            chartData.timestamps.shift();
            chartData.wiredLatency.shift();
            chartData.satelliteLatency.shift();
            chartData.wiredLoss.shift();
            chartData.satelliteLoss.shift();
            chartData.wiredBandwidth.shift();
            chartData.satelliteBandwidth.shift();
            chartData.wiredCost.shift();
            chartData.satelliteCost.shift();
        }
        
        // Update all charts
        updateChart('latency', [chartData.wiredLatency, chartData.satelliteLatency]);
        updateChart('loss', [chartData.wiredLoss, chartData.satelliteLoss]);
        updateChart('bandwidth', [chartData.wiredBandwidth, chartData.satelliteBandwidth]);
        updateChart('cost', [chartData.wiredCost, chartData.satelliteCost]);
    } catch (error) {
        console.error('Error updating charts:', error);
    }
}

// Update individual chart
function updateChart(chartName, datasets) {
    try {
        const chart = charts[chartName];
        if (chart && chart.data) {
            chart.data.labels = [...chartData.timestamps];
            chart.data.datasets[0].data = [...datasets[0]];
            chart.data.datasets[1].data = [...datasets[1]];
            chart.update('none'); // No animation for better performance
        }
    } catch (error) {
        console.error(`Error updating chart ${chartName}:`, error);
    }
}

// Add alert
function addAlert(type, title, message) {
    const alertsContainer = document.getElementById('alertsContainer');
    const alertClass = type === 'warning' ? 'alert-warning' : 
                      type === 'danger' ? 'alert-danger' : 'alert-success';
    
    const alert = document.createElement('div');
    alert.className = `alert ${alertClass}`;
    alert.innerHTML = `
        <div class="alert-icon">
            <i class="fas fa-${type === 'warning' ? 'exclamation-triangle' : 
                              type === 'danger' ? 'times-circle' : 'check-circle'}"></i>
        </div>
        <div class="alert-content">
            <div class="alert-title">${title}</div>
            <div class="alert-message">${message}</div>
        </div>
        <div class="alert-time">${new Date().toLocaleTimeString()}</div>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentNode) {
            alert.parentNode.removeChild(alert);
        }
    }, 5000);
}

// Simulation functions
function startSimulation() {
    if (simulationRunning) {
        console.log('Simulation already running');
        return;
    }
    
    console.log('Starting simulation...');
    simulationRunning = true;
    
    // Start backend simulation
    fetch('/api/start', { method: 'POST' })
        .catch(error => console.log('Backend not available, using local simulation'));
    
    addAlert('success', 'Simulation Started', 'AI Network Stabilization is now monitoring the hybrid network');
    
    simulationInterval = setInterval(() => {
        if (!simulationRunning) return;
        
        fetchNetworkData()
            .then(data => {
                updateMetrics(data);
                updateCharts(data);
            })
            .catch(error => {
                console.error('Error fetching data:', error);
                // Use local data as fallback
                const localData = generateLocalData();
                updateMetrics(localData);
                updateCharts(localData);
            });
        
        // Fetch alerts from backend
        fetch('/api/alerts')
            .then(response => response.json())
            .then(alerts => {
                if (alerts.length > 0) {
                    const latestAlert = alerts[alerts.length - 1];
                    addAlert(latestAlert.type, latestAlert.title, latestAlert.message);
                }
            })
            .catch(error => {
                // Fallback to local alerts
                if (Math.random() < 0.05) { // Reduced frequency
                    addAlert('warning', 'Network Event', 'Satellite handoff detected - switching to optimal path');
                }
            });
    }, 3000); // Increased interval for better performance
}

function stopSimulation() {
    if (!simulationRunning) {
        console.log('Simulation not running');
        return;
    }
    
    console.log('Stopping simulation...');
    simulationRunning = false;
    clearInterval(simulationInterval);
    
    // Stop backend simulation
    fetch('/api/stop', { method: 'POST' })
        .catch(error => console.log('Backend not available'));
    
    addAlert('warning', 'Simulation Stopped', 'AI monitoring has been paused');
}

function resetSimulation() {
    console.log('Resetting simulation...');
    stopSimulation();
    
    // Reset backend
    fetch('/api/reset', { method: 'POST' })
        .catch(error => console.log('Backend not available'));
    
    // Reset data
    chartData = {
        timestamps: [],
        wiredLatency: [],
        satelliteLatency: [],
        wiredLoss: [],
        satelliteLoss: [],
        wiredBandwidth: [],
        satelliteBandwidth: [],
        wiredCost: [],
        satelliteCost: []
    };
    
    // Reset charts
    updateChart('latency', [[], []]);
    updateChart('loss', [[], []]);
    updateChart('bandwidth', [[], []]);
    updateChart('cost', [[], []]);
    
    // Reset metrics
    document.getElementById('activePath').textContent = 'WIRED';
    document.getElementById('wiredLatency').textContent = '32.8 ms';
    document.getElementById('satelliteLatency').textContent = '43.9 ms';
    document.getElementById('aiConfidence').textContent = '60.0%';
    
    addAlert('success', 'Simulation Reset', 'All data has been cleared and system reset');
}

function switchToWired() {
    console.log('Switching to wired...');
    fetch('/api/switch/wired', { method: 'POST' })
        .catch(error => console.log('Backend not available'));
    
    document.getElementById('activePath').textContent = 'WIRED';
    addAlert('success', 'Path Switch', 'Traffic switched to wired network');
}

function switchToSatellite() {
    console.log('Switching to satellite...');
    fetch('/api/switch/satellite', { method: 'POST' })
        .catch(error => console.log('Backend not available'));
    
    document.getElementById('activePath').textContent = 'SATELLITE';
    addAlert('success', 'Path Switch', 'Traffic switched to satellite network');
}

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing dashboard...');
    
    // Initialize components with error handling
    try {
        initializeCharts();
        console.log('Charts initialized');
    } catch (error) {
        console.error('Error initializing charts:', error);
    }
    
    try {
        initializeGlobe();
        console.log('Globe initialized');
    } catch (error) {
        console.error('Error initializing globe:', error);
    }
    
    // Start simulation automatically after a delay
    setTimeout(() => {
        console.log('Auto-starting simulation...');
        startSimulation();
    }, 2000);
});
