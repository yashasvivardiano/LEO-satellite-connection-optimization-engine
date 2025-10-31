# PowerShell script to kill process on port 8501
$port = 8501
Write-Host "Looking for processes on port $port..."

$connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue

if ($connections) {
    $connections | ForEach-Object {
        $processId = $_.OwningProcess
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            Write-Host "Killing process: $($process.ProcessName) (PID: $processId)"
            Stop-Process -Id $processId -Force
        }
    }
    Write-Host "✅ Processes on port $port have been terminated"
} else {
    Write-Host "ℹ️  No processes found on port $port"
}

