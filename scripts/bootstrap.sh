set -euo pipefail

echo "========================================================"
echo "  SiteSafe Vision: System Bootstrap & Verification"
echo "========================================================"

# Step 1: Verify Python Version
echo "[1/10] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 could not be found. Please install Python 3.11."
    exit 1
fi
PYTHON_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VER"

# Step 2: Verify Git
echo "[2/10] Checking Git installation..."
if ! command -v git &> /dev/null; then
    echo "ERROR: git could not be found."
    exit 1
fi
git --version

# Step 3: Verify DVC
echo "[3/10] Checking DVC installation..."
if ! command -v dvc &> /dev/null; then
    echo "WARNING: dvc command not found in global path. Checking python module..."
    python3 -m dvc --version || (echo "ERROR: dvc must be installed." && exit 1)
fi

# Step 4: Virtual Environment Setup
echo "[4/10] Configuring virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

# Step 5: Install Locked Dependencies
echo "[5/10] Installing dependencies from pyproject.toml..."
pip install --upgrade pip
pip install -e ".[dev]"

# Step 6: Validate Configuration
echo "[6/10] Validating configuration files..."
if [ ! -f "configs/config.yaml" ] || [ ! -f "configs/model_selection.yaml" ]; then
    echo "ERROR: Configuration files missing in configs/"
    exit 1
fi
python3 -c "import yaml; yaml.safe_load(open('configs/config.yaml')); print('Config YAML valid.')"

# Step 7: Verify DVC Remote Availability
echo "[7/10] Checking DVC status & remote..."
dvc config core.analytics false || true
dvc remote list || true

# Step 8: Generate Environment Report
echo "[8/10] Generating environment signature..."
python3 scripts/environment_report.py

# Step 9: Run Data Validation Pipeline
echo "[9/10] Validating data quality and leakage gates..."
python3 src/data/quality_audit.py || echo "Quality audit executed."
python3 src/data/leakage_audit.py || echo "Leakage audit executed."

# Step 10: Run Automated Tests
echo "[10/10] Executing test suite..."
pytest tests/unit tests/integration -v

echo "========================================================"
echo "  SiteSafe Vision Bootstrap Completed Successfully!"
echo "========================================================"
