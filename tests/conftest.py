import sys
from pathlib import Path

# Ensure project root is in sys.path during test runs
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.fixtures.create_fixtures import ensure_test_artifacts

# Guarantee baseline test artifacts & models exist before test module collection
ensure_test_artifacts()
