# Activate venv and run the Flask app

if (-not (Test-Path .venv\Scripts\Activate.ps1)) {
    Write-Error "Virtual environment not found. Run setup_env.ps1 first."
    exit 1
}

& .venv\Scripts\Activate.ps1

# Run the app
python app.py