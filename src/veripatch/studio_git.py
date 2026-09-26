"""Structured, repository-scoped Git operations for RAgent Studio."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Any


class GitToolError(RuntimeError):
    """Raised when a structured Git operation cannot be completed safely."""


class StudioGit:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if not (self.root / ".git").exists():
            raise GitToolError("当前工作区不是 Git 仓库")

    @classmethod
    def initialize(cls, root: Path, branch: str = "main") -> dict[str, Any]:
        resolved = root.resolve()
        if not resolved.is_dir():
            raise GitToolError("工作区不存在")
        if (resolved / ".git").exists():
            raise GitToolError("当前工作区已经是 Git 仓库")
        safe_branch = cls._safe_branch(branch)
        completed = subprocess.run(
            ["git", "init", "-b", safe_branch],
            cwd=resolved,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "Git init failed").strip()
            raise GitToolError(detail[-2_000:])
        return {"initialized": True, "branch": safe_branch, "stdout": completed.stdout}

    def _run(
        self, *args: str, timeout: int = 30, allowed_exit_codes: tuple[int, ...] = (0,)
    ) -> dict[str, Any]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(secret in key.casefold() for secret in ("api_key", "token", "secret"))
        }
        completed = subprocess.run(
            ["git", *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            env=env,
        )
        payload = {
            "argv": ["git", *args],
            "exit_code": completed.returncode,
            "stdout": completed.stdout[-40_000:],
            "stderr": completed.stderr[-12_000:],
        }
        if completed.returncode not in allowed_exit_codes:
            detail = (completed.stderr or completed.stdout or "Git command failed").strip()
            raise GitToolError(detail[-2_000:])
        return payload

    def status(self) -> dict[str, Any]:
        payload = self._run("status", "--short", "--branch")
        lines = payload["stdout"].splitlines()
        payload["branch"] = lines[0][3:] if lines and lines[0].startswith("## ") else ""
        payload["changes"] = lines[1:] if lines and lines[0].startswith("## ") else lines
        return payload

    def diff(self, path: str | None = None, revision: str | None = None) -> dict[str, Any]:
        safe_path = self._safe_path(path) if path else None
        if safe_path and not revision:
            untracked = self._run(
                "ls-files", "--others", "--exclude-standard", "--", safe_path
            )["stdout"].splitlines()
            if safe_path in untracked:
                payload = self._run(
                    "diff",
                    "--no-index",
                    "--no-color",
                    "--",
                    os.devnull,
                    safe_path,
                    allowed_exit_codes=(0, 1),
                )
                payload["path"] = path
                payload["revision"] = None
                payload["untracked"] = True
                return payload
        args = ["diff", "--no-ext-diff", "--no-color"]
        if revision:
            args.append(self._safe_revision(revision))
        if path:
            args.extend(["--", safe_path or self._safe_path(path)])
        payload = self._run(*args)
        payload["path"] = path
        payload["revision"] = revision
        return payload

    def log(self, limit: int = 10) -> dict[str, Any]:
        count = max(1, min(limit, 50))
        try:
            payload = self._run(
                "log",
                f"-{count}",
                "--date=iso-strict",
                "--pretty=format:%h%x09%ad%x09%an%x09%s",
            )
        except GitToolError as exc:
            if "does not have any commits" not in str(exc) and "unknown revision" not in str(exc):
                raise
            payload = {"argv": ["git", "log"], "exit_code": 0, "stdout": "", "stderr": ""}
        payload["commits"] = [
            {"sha": parts[0], "date": parts[1], "author": parts[2], "subject": parts[3]}
            for line in payload["stdout"].splitlines()
            if len(parts := line.split("\t", 3)) == 4
        ]
        return payload

    def branches(self, create: str | None = None) -> dict[str, Any]:
        if create:
            if not self.has_commits():
                raise GitToolError("请先完成首次提交，再创建新分支")
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", create):
                raise GitToolError("分支名称不合法")
            if ".." in create or create.endswith(("/", ".")):
                raise GitToolError("分支名称不合法")
            payload = self._run("switch", "-c", create)
            payload["created"] = create
        else:
            payload = self._run("branch", "--format=%(refname:short)")
            payload["created"] = None
        payload["branches"] = [line for line in payload["stdout"].splitlines() if line]
        current = self._run("branch", "--show-current")["stdout"].strip()
        if current and current not in payload["branches"]:
            payload["branches"].insert(0, current)
        payload["current"] = current
        return payload

    def switch(self, branch: str) -> dict[str, Any]:
        if not self.has_commits():
            raise GitToolError("请先完成首次提交，再切换分支")
        safe_branch = self._safe_branch(branch)
        payload = self._run("switch", safe_branch)
        payload["current"] = safe_branch
        return payload

    def has_commits(self) -> bool:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=self.root,
            capture_output=True,
            timeout=10,
            check=False,
        )
        return completed.returncode == 0

    def commit(self, message: str, paths: list[str]) -> dict[str, Any]:
        clean_message = " ".join(message.split()).strip()
        if not clean_message or len(clean_message) > 500:
            raise GitToolError("提交说明不能为空或超过 500 字符")
        safe_paths = list(dict.fromkeys(self._safe_path(path) for path in paths))
        if not safe_paths:
            raise GitToolError("本轮没有可提交的 RAgent 文件改动")
        # `git commit --only` rejects paths that are still untracked. Stage only
        # the explicitly approved RAgent paths first; unrelated staged or
        # untracked user files remain outside the path-limited commit below.
        self._run("add", "--", *safe_paths)
        payload = self._run("commit", "--only", "-m", clean_message, "--", *safe_paths, timeout=60)
        payload["committed_paths"] = safe_paths
        payload["commit"] = self._run("rev-parse", "HEAD")["stdout"].strip()
        return payload

    def restore(self, path: str, revision: str = "HEAD") -> dict[str, Any]:
        safe_path = self._safe_path(path)
        safe_revision = self._safe_revision(revision)
        payload = self._run("restore", "--source", safe_revision, "--worktree", "--", safe_path)
        payload["path"] = safe_path
        payload["revision"] = safe_revision
        return payload

    def _safe_path(self, value: str) -> str:
        supplied = Path(value)
        candidate = (
            supplied.resolve()
            if supplied.is_absolute()
            else (self.root / supplied).resolve()
        )
        if not candidate.is_relative_to(self.root) or candidate == self.root:
            raise GitToolError(f"Git 路径必须位于当前仓库内：{value}")
        relative = candidate.relative_to(self.root)
        if any(part in {".git", ".codex"} for part in relative.parts):
            raise GitToolError(f"禁止直接操作仓库元数据：{value}")
        return relative.as_posix()

    @staticmethod
    def _safe_revision(value: str) -> str:
        revision = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@{}~^+-]{0,199}", revision):
            raise GitToolError("Git revision 不合法")
        return revision

    @staticmethod
    def _safe_branch(value: str) -> str:
        branch = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}", branch):
            raise GitToolError("分支名称不合法")
        if ".." in branch or branch.endswith(("/", ".")):
            raise GitToolError("分支名称不合法")
        return branch
