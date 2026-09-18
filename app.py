import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and execute the Streamlit application for Streamlit Cloud & local Streamlit runners
try:
    import streamlit_app  # noqa: F401
except Exception as e:
    # Fallback to Gradio interface if launched as a standalone script
    if __name__ == "__main__":
        try:
            from demo.app import demo
            demo.launch()
        except Exception:
            raise e
