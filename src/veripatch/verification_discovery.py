"""Read project metadata to propose verification commands without executing them."""

import json
import os
from pathlib import Path

from veripatch.indexing import PythonSymbolIndex


def discover_verification(root: Path, changed: list[str], override: list[str]) -> list[dict]:
    if override:
        return [{"command": override, "source": "manual", "kind": "verification"}]
    root = root.resolve()
    # A frontend subproject owns its scripts; invoke its manager from the repository root.
    if changed and not (root / "package.json").is_file():
        owners = set()
        for name in changed:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                return []
            owner = next(
                (
                    p
                    for p in path.parents
                    if p != root
                    and p.is_relative_to(root)
                    and (p / "package.json").is_file()
                    and (p / "package.json").resolve().is_relative_to(root)
                ),
                None,
            )
            owners.add(owner)
        if len(owners) == 1 and None not in owners:
            owner = owners.pop()
            relative = owner.relative_to(root).as_posix()
            commands = discover_verification(
                owner, [(root / p).resolve().relative_to(owner).as_posix() for p in changed], []
            )
            return [
                {
                    **c,
                    "command": [
                        c["command"][0],
                        "--cwd" if c["command"][0] == "yarn" else "--prefix",
                        relative,
                        *c["command"][1:],
                    ],
                    "source": f"{relative}/{c['source']}",
                }
                for c in commands
                if c["command"][0] in {"npm", "pnpm", "yarn"}
            ]
    candidates = []

    def exists(name):
        path = root / name
        return path.is_file() and path.resolve().is_relative_to(root.resolve())

    def add(command, source, kind="test"):
        candidates.append({"command": command, "source": source, "kind": kind})

    suffixes = {Path(p).suffix.lower() for p in changed}
    if exists("package.json") and (
        not changed or suffixes & {".js", ".jsx", ".ts", ".tsx", ".json", ".css", ".html"}
    ):
        try:
            path = root / "package.json"
            package = (
                json.loads(path.read_text(encoding="utf-8"))
                if path.stat().st_size < 512_000
                else {}
            )
            scripts = package.get("scripts", {}) if isinstance(package, dict) else {}
            if isinstance(scripts, dict):
                manager = (
                    "pnpm" if exists("pnpm-lock.yaml") else "yarn" if exists("yarn.lock") else "npm"
                )
                for name in ("test", "typecheck", "check", "lint", "build"):
                    body = scripts.get(name)
                    if isinstance(body, str) and body.strip() and "no test specified" not in body:
                        add([manager, "run", name], f"package.json:scripts.{name}", name)
        except (OSError, ValueError):
            pass
    if exists("Cargo.toml") and (not changed or suffixes & {".rs", ".toml"}):
        add(["cargo", "test"], "Cargo.toml")
    if exists("go.mod") and (not changed or suffixes & {".go", ".mod"}):
        add(["go", "test", "./..."], "go.mod")
    elif ".go" in suffixes:
        files = [p for p in changed if Path(p).suffix == ".go" and exists(p)]
        if files:
            add(["go", "build", *files], "changed Go files", "build")
    if (
        not changed
        or ".py" in suffixes
        or any(
            Path(p).name in {"pyproject.toml", "pytest.ini", "setup.cfg", "requirements.txt"}
            for p in changed
        )
    ):
        # Discover real test files even when the project has no pytest configuration.
        tests = []
        ignored = {".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist"}
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [
                d for d in dirs if d not in ignored and not (Path(directory) / d).is_symlink()
            ]
            for name in names:
                if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
                    path = Path(directory) / name
                    relative = path.relative_to(root).as_posix()
                    if exists(relative):
                        tests.append(relative)
        tests.sort()
        files = [p for p in changed if Path(p).suffix.lower() == ".py" and exists(p)]
        selected = set()
        dependency_selected = set()
        index = PythonSymbolIndex(root).build() if files and tests else None
        # Prefer AST dependencies, retain conventional names for unimported tests,
        # and use the suite when any changed Python file remains uncovered.
        covered = bool(changed) and len(files) == len(changed)
        for file in files:
            stem = Path(file).stem
            named_matches = {
                t for t in tests if t == file or Path(t).stem in {f"test_{stem}", f"{stem}_test"}
            }
            graph_matches = set(index.related_tests([file])) if index else set()
            dependency_selected.update(graph_matches)
            matches = named_matches | graph_matches
            if stem in {"conftest", "__init__"} or not matches:
                covered = False
            selected.update(matches)
        if tests:
            targets = sorted(selected) if covered and len(selected) <= 12 else []
            add(
                ["python", "-m", "pytest", "-q", *targets],
                (
                    "dependency-related test files"
                    if targets and dependency_selected
                    else "associated test files"
                    if targets
                    else "discovered test suite"
                ),
                "targeted_test" if targets else "test",
            )
        if files:
            add(["python", "-m", "py_compile", *files], "existing changed Python files", "syntax")
    return candidates
