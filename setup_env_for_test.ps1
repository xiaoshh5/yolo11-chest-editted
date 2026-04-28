# setup_env.ps1
# Automates environment setup for the exported project

$ErrorActionPreference = "Stop"

Write-Host ">>> Checking Python..."
try {
    $py = Get-Command python -ErrorAction Stop
    Write-Host "Found Python: $($py.Source)"
} catch {
    Write-Host "Python not found in PATH. Please install Python 3.10+ and add to PATH." -ForegroundColor Red
    # Try to use the included installer?
    if (Test-Path "python-3.11.9-amd64.exe") {
        Write-Host "Found installer: python-3.11.9-amd64.exe. You can run this to install Python." -ForegroundColor Yellow
    }
    exit 1
}

if (-not (Test-Path "venv")) {
    Write-Host ">>> Creating virtual environment (venv)..."
    python -m venv venv
} else {
    Write-Host ">>> Virtual environment 'venv' already exists."
}

Write-Host ">>> Activating venv and installing requirements..."
# Activate script depends on shell, but for this script we just call pip directly from venv
$pip = ".\venv\Scripts\pip.exe"

if (-not (Test-Path $pip)) {
    Write-Host "Error: pip not found at $pip. Venv creation might have failed." -ForegroundColor Red
    exit 1
}

& $pip install --upgrade pip
& $pip install -r requirements.txt

Write-Host ">>> Setup complete!" -ForegroundColor Green
Write-Host "You can now run 'run_app.ps1' to start the application."
Pause
