"""Load the independently built RAgent frontend document."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path


def frontend_root() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "frontend"
    return Path(__file__).resolve().parents[2] / "frontend" / "dist"


@lru_cache(maxsize=1)
def studio_html() -> str:
    """Return the Vite-built frontend without embedding UI code in Python."""
    path = frontend_root() / "index.html"
    if not path.is_file():
        raise RuntimeError("RAgent frontend is not built; run npm --prefix frontend run build")
    return path.read_text(encoding="utf-8")
