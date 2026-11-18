"""Location of the half cell OCP files."""

from __future__ import annotations

import os
from pathlib import Path


def get_ocp_data_root() -> Path:
    """Folder with the two half cell .mat files (OCP_DATA_ROOT overrides it)."""
    env = os.environ.get("OCP_DATA_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "data" / "ocp"
