"""Sandbox distribution layout checks; no installer/EXE is actually executed."""

import hashlib
import importlib.util
import io
import json
import subprocess
from pathlib import Path

import pytest


def builder():
    path = Path(__file__).resolve().parents[1] / "scripts/build_desktop.py"
    spec = importlib.util.spec_from_file_location("sandbox_build", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_assets_include_runtime_node_and_licenses_without_system_install(tmp_path, monkeypatch):
    module = builder()
    source = tmp_path / "sandbox/runtime"
    package = source / "node_modules/@anthropic-ai/sandbox-runtime"
    package.mkdir(parents=True)
    (package / "package.json").write_text(json.dumps({"version": "0.0.78"}), encoding="utf-8")
    (package / "LICENSE").write_text("Apache-2.0 test fixture", encoding="utf-8")
    (source / "bridge.mjs").write_text("// fixture", encoding="utf-8")
    helper = source / "ragent-srt-win.exe"
    helper.write_bytes(b"patched helper fixture")
    patch = tmp_path / "sandbox/patches/windows-acl.patch"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(b"patch fixture")
    (source / "windows-helper.json").write_text(json.dumps({
        "upstream_commit": "6f0ce155ccb136bda33a8a72201fe7f54fe47d9b",
        "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    }))
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/SANDBOX.md").write_text("sandbox limitations", encoding="utf-8")
    tools = tmp_path / "tools"
    tools.mkdir()
    node = tools / "node.exe"
    node.write_bytes(b"fake binary - not executable")
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "v22.22.1\n", "")

    monkeypatch.setattr(
        module.shutil, "which", lambda name: str(node) if name == "node" else "npm.cmd"
    )
    monkeypatch.setattr(module.subprocess, "run", run)
    monkeypatch.setattr(
        module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"Node MIT license fixture")
    )
    staged = module.sandbox_assets(tmp_path)
    assert (staged / "node.exe").read_bytes() == node.read_bytes()
    assert (staged / "bridge.mjs").is_file()
    assert (staged / "ragent-srt-win.exe").read_bytes() == helper.read_bytes()
    assert (staged / "windows-acl.patch").read_bytes() == patch.read_bytes()
    assert (staged / "LICENSE.node.txt").read_bytes() == b"Node MIT license fixture"
    assert (staged / "node_modules/@anthropic-ai/sandbox-runtime/LICENSE").is_file()
    npm = calls[1]
    assert "ci" in npm and "--ignore-scripts" in npm
    assert "--registry=https://registry.npmjs.org" in npm
    assert all(
        "installWindowsSandbox" not in arg and "windows-install" not in arg
        for call in calls
        for arg in call
    )


def test_build_rejects_unsupported_node_version(tmp_path, monkeypatch):
    module = builder()
    monkeypatch.setattr(module.shutil, "which", lambda name: "tool")
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 0, "v18.0.0", ""),
    )
    with pytest.raises(SystemExit, match="22.12"):
        module.sandbox_assets(tmp_path)


@pytest.mark.parametrize("broken", ["binary", "patch", "commit", "missing"])
def test_build_rejects_unverified_native_helper(tmp_path, broken):
    module = builder()
    helper = tmp_path / "ragent-srt-win.exe"
    helper.write_bytes(b"verified helper")
    patch = tmp_path / "windows-acl.patch"
    patch.write_bytes(b"reviewed patch")
    manifest = {
        "upstream_commit": "6f0ce155ccb136bda33a8a72201fe7f54fe47d9b",
        "sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
    }
    if broken == "binary":
        helper.write_bytes(b"different binary")
    elif broken == "patch":
        patch.write_bytes(b"different patch")
    elif broken == "commit":
        manifest["upstream_commit"] = "unexpected"
    else:
        helper.unlink()
    (tmp_path / "windows-helper.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SystemExit, match="helper"):
        module.verify_windows_helper(tmp_path, patch)
