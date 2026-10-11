"""Download and checksum pinned embedding data for offline runtime and EXE builds."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path

from veripatch.embedding_assets import ARTIFACTS, MODEL_ID, REVISION, verify_artifacts


def prepare(directory: Path) -> Path:
    try:
        verify_artifacts(directory)
        valid = True
    except ValueError:
        valid = False
    for name, (size, digest) in (() if valid else ARTIFACTS.items()):
        import hashlib

        target = directory / name
        if target.is_file() and target.stat().st_size == size:
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == digest:
                    continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}"
        print(f"Downloading pinned embedding data: {name}", flush=True)
        # Only these fixed HTTPS artifacts are downloaded, never user-supplied URLs/code.
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        with temporary.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if temporary.stat().st_size != size or actual != digest:
            raise ValueError(f"Invalid embedding artifact: {name}")
        temporary.replace(target)
    verify_artifacts(directory)
    # Include upstream attribution and Apache-2.0 license alongside redistributed weights.
    notices = Path(__file__).resolve().parents[1] / "docs" / "memory-model"
    for name in ("NOTICE.md", "LICENSE"):
        shutil.copy2(notices / name, directory / name)
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path,
                        default=Path(__file__).resolve().parents[1] / "build" / "memory-model")
    args = parser.parse_args()
    print(prepare(args.directory.resolve()))


if __name__ == "__main__":
    main()
