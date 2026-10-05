# APP_MODE=cloud (default) or local. Cloud mode never imports the local-mode module, so it never reaches FastAPI or the full store.
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

MODES = {"cloud": "dashboard.cloud_mode", "local": "dashboard.local_mode"}


def main() -> None:
    mode = os.environ.get("APP_MODE", "cloud").strip().lower()
    st.set_page_config(page_title="H&M Next Best Offer", layout="wide")
    if mode not in MODES:
        st.error(f"APP_MODE must be 'cloud' or 'local', got {mode!r}.")
        st.stop()
    importlib.import_module(MODES[mode]).run()


main()
