"""Build the standalone Windows RAgent executable with PyInstaller."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    if importlib.util.find_spec("mcp") is None:
        raise SystemExit(
            "缺少 MCP SDK。请先运行 `python -m pip install -e .`，再重新打包。"
        )
    subprocess.run(["npm", "run", "build"], cwd=project / "frontend", check=True, shell=True)
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        "RAgent",
        "--icon",
        str(project / "assets" / "ragent.ico"),
        "--paths",
        str(project / "src"),
        "--add-data",
        f"{project / 'examples'};examples",
        "--add-data",
        f"{project / 'assets'};assets",
        "--add-data",
        f"{project / 'frontend' / 'dist'};frontend",
        "--collect-all",
        "webview",
        "--collect-all",
        "pystray",
        "--collect-all",
        "PIL",
        "--collect-all",
        "pytest",
        "--hidden-import",
        "pystray",
        "--collect-submodules",
        "mcp.client",
        "--collect-submodules",
        "mcp.shared",
        "--collect-submodules",
        "keyring.backends",
        str(project / "src" / "veripatch" / "desktop.py"),
    ]
    subprocess.run(command, cwd=project, check=True)


if __name__ == "__main__":
    main()
