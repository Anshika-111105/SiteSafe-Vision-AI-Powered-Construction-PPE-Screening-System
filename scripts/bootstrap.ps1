$ErrorActionPreference = "Stop"

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  SiteSafe Vision: Windows System Bootstrap & Verification" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Step 1: Verify Python
Write-Host "[1/10] Checking Python installation..." -ForegroundColor Yellow
$pyVer = python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
Write-Host "Found Python version: $pyVer" -ForegroundColor Green

# Step 2: Verify Git
Write-Host "[2/10] Checking Git installation..." -ForegroundColor Yellow
git --version

# Step 3: Verify DVC
Write-Host "[3/10] Checking DVC installation..." -ForegroundColor Yellow
dvc --version

# Step 4: Virtual Environment Setup
Write-Host "[4/10] Verifying virtual environment / packages..." -ForegroundColor Yellow
if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv virtual environment..."
    python -m venv .venv
}
if (Test-Path ".venv/Scripts/Activate.ps1") {
    & .venv/Scripts/Activate.ps1
}

# Step 5: Install Locked Dependencies
Write-Host "[5/10] Installing dependencies from pyproject.toml..." -ForegroundColor Yellow
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Step 6: Validate Configuration
Write-Host "[6/10] Validating configuration files..." -ForegroundColor Yellow
if (-not (Test-Path "configs/config.yaml")) {
    Write-Error "ERROR: configs/config.yaml missing."
}
python -c "import yaml; yaml.safe_load(open('configs/config.yaml')); print('Config YAML valid.')"

# Step 7: Verify DVC Remote
Write-Host "[7/10] Checking DVC status & remote..." -ForegroundColor Yellow
dvc config core.analytics false
dvc remote list

# Step 8: Generate Environment Report
Write-Host "[8/10] Generating environment signature..." -ForegroundColor Yellow
python scripts/environment_report.py

# Step 9: Run Data Validation Pipeline
Write-Host "[9/10] Validating data quality and leakage gates..." -ForegroundColor Yellow
if (Test-Path "src/data/quality_audit.py") {
    python src/data/quality_audit.py
}
if (Test-Path "src/data/leakage_audit.py") {
    python src/data/leakage_audit.py
}

# Step 10: Run Automated Tests
Write-Host "[10/10] Executing test suite..." -ForegroundColor Yellow
pytest tests/unit tests/integration -v

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  SiteSafe Vision Bootstrap Completed Successfully!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
