from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from veripatch.studio_git import GitToolError, StudioGit


def git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def repository(tmp_path: Path) -> Path:
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "RAgent Tests")
    git(tmp_path, "config", "user.email", "ragent@example.invalid")
    (tmp_path / "app.py").write_text("value = 1\n", encoding="utf-8")
    git(tmp_path, "add", "app.py")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def test_structured_git_read_tools(tmp_path: Path) -> None:
    root = repository(tmp_path)
    (root / "app.py").write_text("value = 2\n", encoding="utf-8")
    tool = StudioGit(root)

    assert tool.status()["changes"] == [" M app.py"]
    assert "+value = 2" in tool.diff("app.py")["stdout"]
    assert tool.log()["commits"][0]["subject"] == "initial"
    assert tool.branches()["current"]


def test_git_commit_only_uses_explicit_ragent_paths(tmp_path: Path) -> None:
    root = repository(tmp_path)
    (root / "app.py").write_text("value = 2\n", encoding="utf-8")
    (root / "user.txt").write_text("unrelated", encoding="utf-8")

    payload = StudioGit(root).commit("update app", ["app.py"])

    assert payload["committed_paths"] == ["app.py"]
    assert git(root, "show", "--pretty=", "--name-only", "HEAD") == "app.py"
    assert "?? user.txt" in StudioGit(root).status()["changes"]


def test_git_restore_rejects_escape_and_restores_file(tmp_path: Path) -> None:
    root = repository(tmp_path)
    (root / "app.py").write_text("value = 9\n", encoding="utf-8")
    tool = StudioGit(root)

    payload = tool.restore("app.py")

    assert payload["revision"] == "HEAD"
    assert (root / "app.py").read_text(encoding="utf-8") == "value = 1\n"
    with pytest.raises(GitToolError, match="仓库内"):
        tool.diff("../outside.py")


def test_git_switch_changes_existing_branch(tmp_path: Path) -> None:
    root = repository(tmp_path)
    tool = StudioGit(root)
    original = tool.branches()["current"]
    tool.branches("feature/panel")
    tool.switch(original)
    assert tool.branches()["current"] == original
    with pytest.raises(GitToolError, match="分支名称"):
        tool.switch("../unsafe")


def test_git_initialize_creates_unborn_main_branch(tmp_path: Path) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    payload = StudioGit.initialize(root, "main")
    tool = StudioGit(root)
    assert payload["initialized"] is True
    assert tool.status()["branch"] == "No commits yet on main"
    assert tool.branches()["current"] == "main"
    assert tool.branches()["branches"] == ["main"]
    assert tool.log()["commits"] == []
    with pytest.raises(GitToolError, match="先完成首次提交"):
        tool.branches("feature/too-early")
    with pytest.raises(GitToolError, match="先完成首次提交"):
        tool.switch("main")
    with pytest.raises(GitToolError, match="已经是 Git"):
        StudioGit.initialize(root)


def test_git_diff_shows_untracked_file_as_entirely_added(tmp_path: Path) -> None:
    root = tmp_path / "untracked"
    root.mkdir()
    StudioGit.initialize(root)
    (root / "hello.py").write_text('print("hello")\n', encoding="utf-8")
    payload = StudioGit(root).diff("hello.py")
    assert payload["untracked"] is True
    assert '+print("hello")' in payload["stdout"]


def test_git_first_commit_accepts_only_explicit_untracked_path(tmp_path: Path) -> None:
    root = tmp_path / "first-commit"
    root.mkdir()
    StudioGit.initialize(root)
    git(root, "config", "user.name", "RAgent Tests")
    git(root, "config", "user.email", "ragent@example.invalid")
    (root / "hello.py").write_text('print("hello")\n', encoding="utf-8")
    (root / "user-note.txt").write_text("private\n", encoding="utf-8")
    payload = StudioGit(root).commit("initial hello", ["hello.py"])
    assert payload["committed_paths"] == ["hello.py"]
    assert git(root, "show", "--pretty=", "--name-only", "HEAD") == "hello.py"
    assert "?? user-note.txt" in StudioGit(root).status()["changes"]
