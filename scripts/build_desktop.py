"""Build the standalone Windows RAgent executable with PyInstaller."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


def sandbox_assets(project: Path) -> Path:
    """Stage locked runtime + Node and licenses, never provision Windows users."""
    source = project / "sandbox" / "runtime"
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    node = shutil.which("node")
    if not npm or not node:
        raise SystemExit("打包沙箱需要 Node.js 22 LTS 和 npm；EXE 将内置该运行时。")
    version = subprocess.run(
        [node, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(int(v) for v in match.groups()) < (22, 12, 0):
        raise SystemExit("请使用 Node.js >=22.12.0 构建沙箱。")
    subprocess.run(
        [
            npm,
            "ci",
            "--ignore-scripts",
            "--registry=https://registry.npmjs.org",
            "--cache",
            str(project / ".npm-sandbox-cache"),
        ],
        cwd=source,
        check=True,
        shell=False,
    )
    package = source / "node_modules" / "@anthropic-ai" / "sandbox-runtime" / "package.json"
    if json.loads(package.read_text(encoding="utf-8"))["version"] != "0.0.78":
        raise SystemExit("沙箱依赖版本与 bridge 不一致，请核对锁文件。")
    patch = project / "sandbox/patches/windows-acl.patch"
    verify_windows_helper(source, patch)
    staged = project / "build" / "sandbox-runtime"
    staged.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source, staged, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.map", "*.ts", ".cache")
    )
    shutil.copy2(node, staged / "node.exe")
    shutil.copy2(patch, staged / "windows-acl.patch")
    # Node's binary distribution license must accompany the bundled binary.
    license_url = f"https://raw.githubusercontent.com/nodejs/node/{version}/LICENSE"
    with urllib.request.urlopen(license_url, timeout=30) as response:
        license_text = response.read()
    (staged / "LICENSE.node.txt").write_bytes(license_text)
    shutil.copy2(project / "docs" / "SANDBOX.md", staged / "RAgent-sandbox-notes.md")
    return staged


def verify_windows_helper(source: Path, patch: Path) -> None:
    """Reject missing or mismatched native artifacts before packaging."""
    helper = source / "ragent-srt-win.exe"
    metadata = source / "windows-helper.json"
    if not helper.is_file() or not metadata.is_file() or not patch.is_file():
        raise SystemExit("缺少修复后的 Windows helper；请先运行 scripts/build_sandbox_helper.py。")
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if (
        manifest.get("upstream_commit") != "6f0ce155ccb136bda33a8a72201fe7f54fe47d9b"
        or hashlib.sha256(helper.read_bytes()).hexdigest() != manifest.get("sha256")
        or hashlib.sha256(patch.read_bytes()).hexdigest() != manifest.get("patch_sha256")
    ):
        raise SystemExit("Windows helper/源码补丁校验失败；请重新构建，不打包旧 helper。")


def main() -> None:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=project / "dist",
                        help="输出目录；旧版正在运行时可另存新版，避免强制退出")
    args = parser.parse_args()
    if importlib.util.find_spec("mcp") is None:
        raise SystemExit("缺少 MCP SDK。请先运行 `python -m pip install -e .`，再重新打包。")
    if importlib.util.find_spec("yaml") is None:
        raise SystemExit("缺少 PyYAML。请先运行 `python -m pip install -e .`，再重新打包。")
    sandbox = sandbox_assets(project)
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
        "--distpath",
        str(args.dist_dir.resolve()),
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
        "--add-data",
        f"{sandbox};sandbox/runtime",
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
