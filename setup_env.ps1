r# Run this in PowerShell to create a virtual environment and install dependencies

# Check for Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is not installed or not on PATH. Install Python 3.10+ and enable 'Add to PATH' during installation."
    exit 1
}

# Create venv
python -m venv .venv

# Activate venv
& .venv\Scripts\Activate.ps1

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

Write-Output "Virtualenv created and dependencies installed. Activate with: & .venv\Scripts\Activate.ps1"