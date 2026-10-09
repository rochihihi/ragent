"""Rebuild the pinned, minimally patched Windows helper. Never installs it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

UPSTREAM = "https://github.com/anthropics/sandbox-runtime.git"
COMMIT = "6f0ce155ccb136bda33a8a72201fe7f54fe47d9b"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cargo", default="cargo")
    parser.add_argument("--target", default="x86_64-pc-windows-msvc")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    source = project / "build/helper-source"
    patch = project / "sandbox/patches/windows-acl.patch"
    if not source.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                "v0.0.78",
                "--filter=blob:none",
                "--sparse",
                UPSTREAM,
                str(source),
            ],
            check=True,
        )
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "sparse-checkout",
                "set",
                "vendor/srt-win-src",
                "test/fixtures/tls-terminate",
            ],
            check=True,
        )
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != COMMIT:
        raise SystemExit("Upstream commit mismatch; refusing to build a different helper.")
    command = ["git", "-C", str(source), "apply", "--ignore-space-change"]
    if subprocess.run([*command, "--check", str(patch)], capture_output=True).returncode == 0:
        subprocess.run([*command, str(patch)], check=True)
    elif subprocess.run(
        [*command, "--reverse", "--check", str(patch)], capture_output=True
    ).returncode:
        raise SystemExit("Patch mismatch; existing source left unchanged.")
    diff = subprocess.check_output(
        ["git", "-C", str(source), "diff", "--no-ext-diff", "HEAD"],
        text=True,
        encoding="utf-8",
    )
    if diff.splitlines() != patch.read_text(encoding="utf-8").splitlines():
        raise SystemExit("Unexpected source changes; refusing to build an unaudited helper.")
    manifest = source / "vendor/srt-win-src/Cargo.toml"
    environment = dict(os.environ)
    target_dir = project / "build/helper-target"
    environment["CARGO_TARGET_DIR"] = str(target_dir)
    subprocess.run(
        [
            args.cargo,
            "build",
            "--manifest-path",
            str(manifest),
            "--locked",
            "--release",
            "--target",
            args.target,
        ],
        env=environment,
        check=True,
    )
    subprocess.run(
        [
            args.cargo,
            "test",
            "--manifest-path",
            str(manifest),
            "--locked",
            "--release",
            "--target",
            args.target,
            "acl::tests",
            "--",
            "--test-threads=1",
        ],
        env=environment,
        check=True,
    )
    binary = target_dir / args.target / "release/srt-win.exe"
    runtime = project / "sandbox/runtime"
    shutil.copy2(binary, runtime / "ragent-srt-win.exe")
    metadata = {
        "upstream": UPSTREAM.removesuffix(".git"),
        "upstream_commit": COMMIT,
        "upstream_tag": "v0.0.78",
        "patch": "ragent-windows-acl-v1",
        "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(),
        "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "target": args.target,
    }
    (runtime / "windows-helper.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Built and verified Windows helper; no accounts/firewall settings changed.")


if __name__ == "__main__":
    main()
