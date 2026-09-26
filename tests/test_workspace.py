from pathlib import Path

import pytest

from veripatch.domain import FileEdit
from veripatch.workspace import EditConflictError, SafeWorkspace, WorkspaceSecurityError


def test_workspace_blocks_path_traversal(tmp_path: Path) -> None:
    workspace = SafeWorkspace(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        workspace.read("../secret.txt")


def test_workspace_reads_an_explicitly_approved_external_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "diagnostics"
    external.mkdir()
    log = external / "codex.log"
    log.write_text("startup failed\n", encoding="utf-8")
    workspace = SafeWorkspace(repository, approved_roots=[external])
    assert "startup failed" in workspace.read(str(log))["content"]


def test_external_read_grant_does_not_grant_write(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "note.txt"
    target.write_text("before\n", encoding="utf-8")
    workspace = SafeWorkspace(repository, approved_roots=[external])

    with pytest.raises(WorkspaceSecurityError, match="write permission"):
        workspace.apply_edits(
            [FileEdit(path=str(target), old_text="before", new_text="after")],
            protect_tests=False,
        )


def test_external_write_grant_is_limited_to_approved_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    target = external / "note.txt"
    target.write_text("before\n", encoding="utf-8")
    workspace = SafeWorkspace(
        repository,
        approved_roots=[external],
        approved_write_roots=[external],
    )

    workspace.apply_edits(
        [FileEdit(path=str(target), old_text="before", new_text="after")],
        protect_tests=False,
    )
    assert target.read_text(encoding="utf-8") == "after\n"


def test_workspace_blocks_test_tampering(tmp_path: Path) -> None:
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_app.py").write_text("assert False\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        workspace.apply_edits(
            [FileEdit(path="tests/test_app.py", old_text="False", new_text="True")]
        )


def test_workspace_applies_exact_edit_and_produces_diff(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text("answer = 41\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    changed = workspace.apply_edits(
        [FileEdit(path="app.py", old_text="answer = 41", new_text="answer = 42")]
    )
    assert changed == ["app.py"]
    assert source.read_text(encoding="utf-8") == "answer = 42\n"
    assert "-answer = 41" in workspace.diff()
    assert "+answer = 42" in workspace.diff()


def test_workspace_applies_multifile_unified_patch_atomically(tmp_path: Path) -> None:
    (tmp_path / "one.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "two.py").write_text("name = 'old'\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)

    changed = workspace.apply_patch(
        "--- a/one.py\n"
        "+++ b/one.py\n"
        "@@ -1 +1 @@\n"
        "-value = 1\n"
        "+value = 2\n"
        "--- a/two.py\n"
        "+++ b/two.py\n"
        "@@ -1 +1 @@\n"
        "-name = 'old'\n"
        "+name = 'new'\n"
    )

    assert changed == ["one.py", "two.py"]
    assert (tmp_path / "one.py").read_text(encoding="utf-8") == "value = 2\n"
    assert (tmp_path / "two.py").read_text(encoding="utf-8") == "name = 'new'\n"


def test_workspace_patch_rejects_path_escape(tmp_path: Path) -> None:
    workspace = SafeWorkspace(tmp_path)
    with pytest.raises(WorkspaceSecurityError):
        workspace.apply_patch("--- a/../outside.py\n+++ b/../outside.py\n@@ -1 +1 @@\n-old\n+new\n")


def test_workspace_creates_new_file_and_tracks_diff(tmp_path: Path) -> None:
    workspace = SafeWorkspace(tmp_path)

    created = workspace.create_file("src/new_module.py", "answer = 42\n")

    assert created == "src/new_module.py"
    assert (tmp_path / created).read_text(encoding="utf-8") == "answer = 42\n"
    assert "+answer = 42" in workspace.diff()
    with pytest.raises(EditConflictError, match="already exists"):
        workspace.create_file(created, "replacement")


def test_multi_file_edit_rolls_back_when_a_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    real_atomic_write = workspace._atomic_write
    calls = 0

    def flaky_atomic_write(path: Path, content: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated disk error")
        real_atomic_write(path, content)

    monkeypatch.setattr(workspace, "_atomic_write", flaky_atomic_write)
    with pytest.raises(OSError, match="simulated disk error"):
        workspace.apply_edits(
            [
                FileEdit(path="first.py", old_text="1", new_text="10"),
                FileEdit(path="second.py", old_text="2", new_text="20"),
            ]
        )
    assert first.read_text(encoding="utf-8") == "value = 1\n"
    assert second.read_text(encoding="utf-8") == "value = 2\n"
    assert workspace.diff() == ""
    assert workspace.original_contents == {}


def test_prepared_edit_recovers_before_after_and_mixed_states(tmp_path: Path) -> None:
    first = tmp_path / "first.py"
    second = tmp_path / "second.py"
    first.write_text("value = 1\n", encoding="utf-8")
    second.write_text("value = 2\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    transaction = workspace.prepare_edits(
        [
            FileEdit(path="first.py", old_text="1", new_text="10"),
            FileEdit(path="second.py", old_text="2", new_text="20"),
        ]
    )

    assert workspace.apply_prepared(transaction) == ["first.py", "second.py"]
    assert first.read_text(encoding="utf-8") == "value = 10\n"
    assert workspace.apply_prepared(transaction) == ["first.py", "second.py"]

    first.write_text(transaction.files[0].after_text, encoding="utf-8")
    second.write_text(transaction.files[1].before_text, encoding="utf-8")
    workspace.apply_prepared(transaction)
    assert second.read_text(encoding="utf-8") == "value = 20\n"


def test_prepared_edit_rejects_unknown_workspace_drift(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text("answer = 41\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    transaction = workspace.prepare_edits([FileEdit(path="app.py", old_text="41", new_text="42")])
    source.write_text("answer = 99\n", encoding="utf-8")
    with pytest.raises(EditConflictError, match="drift"):
        workspace.apply_prepared(transaction)


def test_workspace_rejects_ambiguous_edit_and_invalid_reads(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text("same\nsame\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    with pytest.raises(EditConflictError, match="found 2"):
        workspace.prepare_edits([FileEdit(path="app.py", old_text="same", new_text="new")])
    with pytest.raises(ValueError, match="before"):
        workspace.read("app.py", 3, 2)
    with pytest.raises(ValueError, match="empty"):
        workspace.search("  ")


def test_empty_old_text_only_replaces_an_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "app.py"
    source.write_text("", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)
    edit = FileEdit(path="app.py", old_text="", new_text="value = 1\n")

    assert workspace.apply_edits([edit]) == ["app.py"]
    assert source.read_text(encoding="utf-8") == "value = 1\n"
    with pytest.raises(EditConflictError, match="Empty old_text"):
        workspace.prepare_edits([edit])


def test_structured_move_copy_and_delete_are_tracked_in_diff(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)

    copied = workspace.copy_file("source.py", "nested/copied.py")
    moved = workspace.move_file("source.py", "renamed.py")
    deleted = workspace.delete_path("nested/copied.py")

    assert copied == {"source": "source.py", "destination": "nested/copied.py"}
    assert moved == {"source": "source.py", "destination": "renamed.py"}
    assert deleted["kind"] == "file"
    assert not (tmp_path / "source.py").exists()
    assert (tmp_path / "renamed.py").read_text(encoding="utf-8") == "value = 1\n"
    diff = workspace.diff()
    assert "--- a/source.py" in diff
    assert "+++ b/renamed.py" in diff


def test_structured_delete_refuses_non_empty_directory_and_metadata(tmp_path: Path) -> None:
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "keep.txt").write_text("keep", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("secret", encoding="utf-8")
    workspace = SafeWorkspace(tmp_path)

    with pytest.raises(WorkspaceSecurityError, match="empty directories"):
        workspace.delete_path("folder")
    with pytest.raises(WorkspaceSecurityError, match="Protected repository metadata"):
        workspace.delete_path(".git/config")
