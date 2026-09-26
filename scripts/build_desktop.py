"""Build the standalone Windows RAgent executable with PyInstaller."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project = Path(__file__).resolve().parents[1]
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
        "--collect-submodules",
        "keyring.backends",
        str(project / "src" / "veripatch" / "desktop.py"),
    ]
    subprocess.run(command, cwd=project, check=True)
    mcp_command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--console",
        "--name",
        "RAgent-MCP",
        "--paths",
        str(project / "src"),
        str(project / "src" / "veripatch" / "mcp_entry.py"),
    ]
    subprocess.run(mcp_command, cwd=project, check=True)


if __name__ == "__main__":
    main()
