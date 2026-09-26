"""Filesystem tools with strict repository containment and atomic edits."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any
from uuid import uuid4

from veripatch.domain import EditTransaction, FileEdit, PreparedFileChange
from veripatch.indexing import IGNORED_DIRECTORIES


class WorkspaceSecurityError(ValueError):
    """Raised when a requested operation crosses a repository boundary."""


class EditConflictError(ValueError):
    """Raised when an exact edit cannot be applied unambiguously."""


def _is_test_path(relative: Path) -> bool:
    lowered = [part.casefold() for part in relative.parts]
    return (
        "tests" in lowered
        or relative.name.startswith("test_")
        or relative.name.endswith("_test.py")
    )


class SafeWorkspace:
    def __init__(
        self,
        root: Path,
        *,
        max_file_bytes: int = 512_000,
        approved_roots: list[Path] | None = None,
        approved_write_roots: list[Path] | None = None,
    ) -> None:
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError(f"Repository root does not exist: {self.root}")
        self.max_file_bytes = max_file_bytes
        self.approved_roots = [path.resolve() for path in (approved_roots or [])]
        self.approved_write_roots = [path.resolve() for path in (approved_write_roots or [])]
        self._original_contents: dict[str, str] = {}

    @property
    def original_contents(self) -> dict[str, str]:
        return dict(self._original_contents)

    def load_original_contents(self, originals: dict[str, str]) -> None:
        for relative_name in originals:
            self.resolve(relative_name)
        self._original_contents = dict(originals)

    def resolve(self, relative_path: str) -> Path:
        if not relative_path or "\x00" in relative_path:
            raise WorkspaceSecurityError("Path cannot be empty or contain NUL")
        supplied = Path(relative_path)
        candidate = (
            supplied.resolve() if supplied.is_absolute() else (self.root / supplied).resolve()
        )
        roots = [self.root, *self.approved_roots]
        if not any(candidate == root or candidate.is_relative_to(root) for root in roots):
            raise WorkspaceSecurityError(f"Path requires user permission: {relative_path}")
        return candidate

    def _read_text(self, path: Path) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size > self.max_file_bytes:
            raise ValueError(f"File exceeds {self.max_file_bytes} bytes: {path}")
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Only UTF-8 text files are supported: {path}") from exc

    def _assert_write_allowed(self, path: Path) -> None:
        roots = [self.root, *self.approved_write_roots]
        if not any(path == root or path.is_relative_to(root) for root in roots):
            raise WorkspaceSecurityError(f"Path requires write permission: {path}")

    def read(self, relative_path: str, start_line: int = 1, end_line: int = 240) -> dict[str, Any]:
        if end_line < start_line:
            raise ValueError("end_line cannot be before start_line")
        end_line = min(end_line, start_line + 399)
        path = self.resolve(relative_path)
        text = self._read_text(path)
        lines = text.splitlines()
        selected = lines[start_line - 1 : end_line]
        numbered = "\n".join(
            f"{line_number:>5}: {line}"
            for line_number, line in enumerate(selected, start=start_line)
        )
        return {
            "path": (
                path.relative_to(self.root).as_posix()
                if path.is_relative_to(self.root)
                else str(path)
            ),
            "start_line": start_line,
            "end_line": min(end_line, len(lines)),
            "content": numbered,
            "total_lines": len(lines),
        }

    def search(self, query: str, *, limit: int = 30) -> list[dict[str, Any]]:
        needle = query.casefold().strip()
        if not needle:
            raise ValueError("Search query cannot be empty")
        results: list[dict[str, Any]] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(self.root) if path.is_relative_to(self.root) else path
            if any(part in IGNORED_DIRECTORIES for part in relative.parts):
                continue
            if path.stat().st_size > self.max_file_bytes:
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeDecodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line.casefold():
                    results.append(
                        {
                            "path": relative.as_posix(),
                            "line": line_number,
                            "content": line[:500],
                        }
                    )
                    if len(results) >= limit:
                        return results
        return results

    @staticmethod
    def _digest(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def prepare_edits(
        self, edits: list[FileEdit], *, protect_tests: bool = True
    ) -> EditTransaction:
        if not edits:
            raise ValueError("At least one edit is required")
        pending: dict[Path, str] = {}
        action_originals: dict[Path, str] = {}

        for edit in edits:
            path = self.resolve(edit.path)
            self._assert_write_allowed(path)
            relative = path.relative_to(self.root) if path.is_relative_to(self.root) else path
            if any(part in {".git", ".github", ".codex"} for part in relative.parts):
                raise WorkspaceSecurityError(f"Protected repository metadata: {edit.path}")
            if protect_tests and _is_test_path(relative):
                raise WorkspaceSecurityError(f"Test files are read-only: {edit.path}")
            current = pending.get(path, self._read_text(path))
            action_originals.setdefault(path, current)
            if not edit.old_text and current:
                raise EditConflictError(
                    f"Empty old_text can only edit an empty file: {edit.path}"
                )
            occurrences = current.count(edit.old_text)
            if occurrences != 1:
                raise EditConflictError(
                    f"Expected exactly one occurrence in {edit.path}, found {occurrences}"
                )
            pending[path] = current.replace(edit.old_text, edit.new_text, 1)

        files = [
            PreparedFileChange(
                path=(
                    path.relative_to(self.root).as_posix()
                    if path.is_relative_to(self.root)
                    else str(path)
                ),
                before_text=action_originals[path],
                after_text=content,
                before_sha256=self._digest(action_originals[path]),
                after_sha256=self._digest(content),
            )
            for path, content in sorted(pending.items(), key=lambda item: item[0].as_posix())
        ]
        for file in files:
            self._original_contents.setdefault(file.path, file.before_text)
        return EditTransaction(transaction_id=uuid4().hex, edits=edits, files=files)

    def apply_prepared(self, transaction: EditTransaction) -> list[str]:
        statuses: dict[Path, str] = {}
        files_by_path: dict[Path, PreparedFileChange] = {}
        for file in transaction.files:
            path = self.resolve(file.path)
            current = self._read_text(path)
            current_digest = self._digest(current)
            if current_digest == file.before_sha256 and current == file.before_text:
                statuses[path] = "before"
            elif current_digest == file.after_sha256 and current == file.after_text:
                statuses[path] = "after"
            else:
                raise EditConflictError(f"Workspace drift detected for {file.path}")
            files_by_path[path] = file

        if statuses and set(statuses.values()) == {"after"}:
            return sorted(file.path for file in transaction.files)

        if "after" in statuses.values():
            for path, file in files_by_path.items():
                self._atomic_write(path, file.before_text)

        written: list[Path] = []
        try:
            for path, file in files_by_path.items():
                self._atomic_write(path, file.after_text)
                written.append(path)
        except Exception:
            for written_path in reversed(written):
                self._atomic_write(written_path, files_by_path[written_path].before_text)
            raise
        return sorted(file.path for file in transaction.files)

    def apply_edits(self, edits: list[FileEdit], *, protect_tests: bool = True) -> list[str]:
        baseline = dict(self._original_contents)
        try:
            transaction = self.prepare_edits(edits, protect_tests=protect_tests)
            return self.apply_prepared(transaction)
        except Exception:
            self._original_contents = baseline
            raise

    def apply_patch(self, patch: str, *, protect_tests: bool = True) -> list[str]:
        """Apply a conventional unified diff through the atomic exact-edit transaction."""
        edits = self._patch_edits(patch)
        return self.apply_edits(edits, protect_tests=protect_tests)

    def _patch_edits(self, patch: str) -> list[FileEdit]:
        if not patch.strip():
            raise ValueError("Patch cannot be empty")
        lines = patch.splitlines(keepends=True)
        edits: list[FileEdit] = []
        current_path: str | None = None
        index = 0
        while index < len(lines):
            line = lines[index]
            if line.startswith("--- "):
                old_name = line[4:].strip().split("\t", 1)[0]
                index += 1
                if index >= len(lines) or not lines[index].startswith("+++ "):
                    raise ValueError("Patch file header is missing +++")
                new_name = lines[index][4:].strip().split("\t", 1)[0]
                if old_name == "/dev/null" or new_name == "/dev/null":
                    raise ValueError(
                        "Use create or an explicit delete command for new/deleted files"
                    )
                old_path = old_name[2:] if old_name.startswith("a/") else old_name
                new_path = new_name[2:] if new_name.startswith("b/") else new_name
                if old_path != new_path:
                    raise ValueError("Patch renames are not supported")
                self.resolve(new_path)
                current_path = new_path
                index += 1
                continue
            if line.startswith("@@"):
                if current_path is None:
                    raise ValueError("Patch hunk has no file header")
                index += 1
                before: list[str] = []
                after: list[str] = []
                while index < len(lines) and not lines[index].startswith(("@@", "--- ")):
                    hunk_line = lines[index]
                    if hunk_line.startswith("\\ No newline at end of file"):
                        index += 1
                        continue
                    if not hunk_line or hunk_line[0] not in " +-":
                        raise ValueError(f"Invalid patch hunk line: {hunk_line[:80]}")
                    marker, content = hunk_line[0], hunk_line[1:]
                    if marker in " -":
                        before.append(content)
                    if marker in " +":
                        after.append(content)
                    index += 1
                old_text = "".join(before)
                new_text = "".join(after)
                if not old_text:
                    raise ValueError("Insertion-only hunks require surrounding context")
                edits.append(FileEdit(path=current_path, old_text=old_text, new_text=new_text))
                continue
            index += 1
        if not edits:
            raise ValueError("Patch contains no applicable hunks")
        return edits

    def create_file(self, relative_path: str, content: str) -> str:
        path = self.resolve(relative_path)
        self._assert_write_allowed(path)
        relative = path.relative_to(self.root)
        if any(part in {".git", ".github", ".codex"} for part in relative.parts):
            raise WorkspaceSecurityError(f"Protected repository metadata: {relative_path}")
        if path.exists():
            raise EditConflictError(f"File already exists: {relative_path}")
        if len(content.encode("utf-8")) > self.max_file_bytes:
            raise ValueError(f"File exceeds {self.max_file_bytes} bytes: {relative_path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._original_contents.setdefault(relative.as_posix(), "")
        self._atomic_write(path, content)
        return relative.as_posix()

    def move_file(self, source: str, destination: str) -> dict[str, str]:
        source_path, source_relative = self._safe_file_operation_path(source)
        destination_path, destination_relative = self._safe_file_operation_path(
            destination, require_existing=False
        )
        if destination_path.exists():
            raise EditConflictError(f"Destination already exists: {destination}")
        content = self._read_text(source_path)
        self._original_contents.setdefault(source_relative, content)
        self._original_contents.setdefault(destination_relative, "")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(destination_path))
        return {"source": source_relative, "destination": destination_relative}

    def copy_file(self, source: str, destination: str) -> dict[str, str]:
        source_path, source_relative = self._safe_file_operation_path(source)
        destination_path, destination_relative = self._safe_file_operation_path(
            destination, require_existing=False
        )
        if destination_path.exists():
            raise EditConflictError(f"Destination already exists: {destination}")
        content = self._read_text(source_path)
        self._original_contents.setdefault(destination_relative, "")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(destination_path, content)
        return {"source": source_relative, "destination": destination_relative}

    def delete_path(self, relative_path: str) -> dict[str, Any]:
        path = self.resolve(relative_path)
        self._assert_write_allowed(path)
        relative = path.relative_to(self.root)
        self._assert_file_operation_metadata(relative)
        if path.is_dir():
            if any(path.iterdir()):
                raise WorkspaceSecurityError(
                    "Structured delete only removes empty directories; "
                    "non-empty directory deletion requires explicit command approval"
                )
            path.rmdir()
            result = {"path": relative.as_posix(), "kind": "empty_directory"}
        else:
            content = self._read_text(path)
            self._original_contents.setdefault(relative.as_posix(), content)
            path.unlink()
            result = {"path": relative.as_posix(), "kind": "file", "bytes": len(content.encode())}
        if os.path.lexists(path):
            raise OSError(f"Deletion postcondition failed: {relative.as_posix()}")
        return {**result, "postcondition": "absent"}

    def _safe_file_operation_path(
        self, relative_path: str, *, require_existing: bool = True
    ) -> tuple[Path, str]:
        path = self.resolve(relative_path)
        self._assert_write_allowed(path)
        relative = path.relative_to(self.root)
        self._assert_file_operation_metadata(relative)
        if require_existing and not path.is_file():
            raise FileNotFoundError(path)
        return path, relative.as_posix()

    @staticmethod
    def _assert_file_operation_metadata(relative: Path) -> None:
        if any(part in {".git", ".github", ".codex"} for part in relative.parts):
            raise WorkspaceSecurityError(f"Protected repository metadata: {relative.as_posix()}")

    def _atomic_write(self, path: Path, content: str) -> None:
        handle, temporary_name = tempfile.mkstemp(
            prefix=".veripatch-", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
                stream.write(content)
            os.replace(temporary_name, path)
        except Exception:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise

    def diff(self) -> str:
        chunks: list[str] = []
        for relative_name in sorted(self._original_contents):
            original = self._original_contents[relative_name]
            path = self.resolve(relative_name)
            current = self._read_text(path) if path.exists() else ""
            chunks.extend(
                difflib.unified_diff(
                    original.splitlines(keepends=True),
                    current.splitlines(keepends=True),
                    fromfile=f"a/{relative_name}",
                    tofile=f"b/{relative_name}",
                )
            )
        return "".join(chunks)

    def current_hashes(self) -> dict[str, str]:
        return {
            relative_name: self._digest(self._read_text(self.resolve(relative_name)))
            for relative_name in self._original_contents
        }

    def assert_hashes(self, expected: dict[str, str]) -> None:
        for relative_name, expected_digest in expected.items():
            actual = self._digest(self._read_text(self.resolve(relative_name)))
            if actual != expected_digest:
                raise EditConflictError(f"Workspace drift detected for {relative_name}")

    def restore(self) -> None:
        for relative_name, content in self._original_contents.items():
            self._atomic_write(self.resolve(relative_name), content)
