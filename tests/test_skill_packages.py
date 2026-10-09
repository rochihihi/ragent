"""Packages, trust controls, real API round trips, and progressive loading."""

import asyncio
import base64
import io
import os
import stat
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from veripatch import studio_skills as skills
from veripatch.config import Settings
from veripatch.studio_agent import StudioAgent, _file_tree
from veripatch.studio_api import create_studio_router
from veripatch.studio_domain import StudioAction, StudioDecision, StudioReply, StudioSession
from veripatch.studio_store import StudioStore

CONTENT = (
    "---\nname: sales\ndescription: >-\n  分析销售报表，\n  统计金额和商品排名。\n"
    "compatibility: Python 3.11+\nmetadata:\n  version: '1.0'\n---\n只分析实际销售数据。\n"
)


def package():
    return {
        "SKILL.md": CONTENT.encode(),
        "scripts/analyze.py": b"print('example')",
        "references/rules.md": "金额为每行总金额。".encode(),
        "assets/logo.bin": b"\0\xff",
    }


def encoded(files):
    return [{"path": name, "data": base64.b64encode(data).decode()} for name, data in files.items()]


def archive(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
    return base64.b64encode(stream.getvalue()).decode()


def session(tmp_path, **kwargs):
    return StudioSession(
        session_id="skills",
        repo_root=str(tmp_path),
        provider="deepseek",
        model="test",
        reasoning_effort="low",
        **kwargs,
    )


def test_yaml_multiline_and_metadata():
    item = skills.parse(CONTENT)
    assert item["description"] == "分析销售报表， 统计金额和商品排名。"
    assert item["compatibility"] == "Python 3.11+"
    assert item["metadata"]["version"] == "1.0"


@pytest.mark.parametrize(
    "frontmatter",
    [
        "name: sales\nname: other\ndescription: test",
        "name: sales\ndescription: !!python/object/apply:os.system ['echo bad']",
        "name: sales\ndescription: 123",
        "name: sales\ndescription: ''",
        "name: sales\ndescription: &a test\nmetadata: {x: *a}",
        "name: sales\ndescription: test\nmetadata: {x: 1}",
        "name: sales\ndescription: test\nx: " + "[" * 25 + "0" + "]" * 25,
    ],
)
def test_unsafe_or_invalid_yaml(frontmatter):
    with pytest.raises(ValueError):
        skills.parse(f"---\n{frontmatter}\n---\nbody")


@pytest.mark.parametrize("kind", ["folder", "zip", "flat"])
def test_package_roundtrip(tmp_path, kind):
    raw = package()
    if kind == "flat":
        decoded = skills.decode_package(files=encoded(raw))
    else:
        wrapped = {f"sales/{name}": data for name, data in raw.items()}
        decoded = skills.decode_package(
            **({"archive": archive(wrapped)} if kind == "zip" else {"files": encoded(wrapped)})
        )
    before = skills.preview(str(tmp_path), decoded)
    assert before["existing_version"] is None
    assert not (tmp_path / ".agents").exists()
    result = skills.install_package(str(tmp_path), decoded)
    assert result["name"] == "sales"
    assert len(result["files"]) == 4
    assert (Path(result["base_directory"]) / "scripts/analyze.py").read_bytes() == raw[
        "scripts/analyze.py"
    ]
    assert "content" not in skills.listing(str(tmp_path))["items"][0]
    assert skills.resource(str(tmp_path), "sales", "assets/logo.bin")["kind"] == "binary"
    assert (
        skills.resource(str(tmp_path), "sales", "references/rules.md")["content"]
        == "金额为每行总金额。"
    )


@pytest.mark.parametrize(
    "badpath",
    [
        "../outside",
        "/absolute",
        "C:/outside",
        "a/../../x",
        "a\\..\\x",
        "a:stream",
        "aux.txt",
        "a/CON",
        "a./b",
        "a /b",
        "a//b",
        "a/./b",
        "a/\0b",
    ],
)
@pytest.mark.parametrize("kind", ["folder", "zip"])
def test_reject_escaping_paths(tmp_path, badpath, kind):
    raw = {"SKILL.md": CONTENT.encode(), badpath: b"bad"}
    zip_data = None
    if kind == "zip":
        raw = {k.replace("\0", "Z"): v for k, v in raw.items()}
        binary = base64.b64decode(archive(raw))
        if "\0" in badpath:
            binary = binary.replace(b"a/Zb", b"a/\0b")
        zip_data = base64.b64encode(binary).decode()
    with pytest.raises(ValueError):
        skills.decode_package(
            **({"archive": zip_data} if kind == "zip" else {"files": encoded(raw)})
        )
    assert not (tmp_path / ".agents").exists()


def test_reject_zip_symlink_and_duplicate():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as bundle:
        info = zipfile.ZipInfo("script")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(info, "../../secret")
    with pytest.raises(ValueError, match="链接"):
        skills.decode_package(archive=base64.b64encode(stream.getvalue()).decode())
    files = encoded(package())
    files.append({"path": "skill.md", "data": base64.b64encode(b"bad").decode()})
    with pytest.raises(ValueError, match="重复路径"):
        skills.decode_package(files=files)


@pytest.mark.parametrize(
    "extras",
    [
        {"scripts": b"file", "scripts/x.py": b"code"},
        {"Scripts/a.py": b"a", "scripts/b.py": b"b"},
        {"other/SKILL.md": CONTENT.encode()},
    ],
)
def test_conflicting_package_paths(extras):
    raw = {"sales/SKILL.md": CONTENT.encode(), **extras}
    with pytest.raises(ValueError):
        skills.decode_package(files=encoded(raw))


def test_size_and_count_limits():
    with pytest.raises(ValueError):
        skills.decode_package(
            files=encoded({"SKILL.md": CONTENT.encode(), "big": b"x" * (skills.MAX_FILE_BYTES + 1)})
        )
    with pytest.raises(ValueError):
        skills.decode_package(files=encoded({str(i): b"x" for i in range(201)}))
    with pytest.raises(ValueError):
        skills.decode_package(
            archive=archive(
                {"SKILL.md": CONTENT.encode(), "big": b"x" * (skills.MAX_FILE_BYTES + 1)}
            )
        )
    with pytest.raises(ValueError):
        skills.decode_package(content=CONTENT, files=[])


def test_update_conflict_backup_delete_and_rollback(tmp_path):
    first = skills.install_package(str(tmp_path), package())
    changed = {**package(), "SKILL.md": CONTENT.replace("实际", "真实").encode()}
    with pytest.raises(FileExistsError):
        skills.install_package(str(tmp_path), changed)
    with pytest.raises(skills.SkillConflictError):
        skills.install_package(str(tmp_path), changed, replace=True, expected_version="0" * 64)
    result = skills.install_package(
        str(tmp_path), changed, replace=True, expected_version=first["version"]
    )
    assert (Path(result["backup"]) / "SKILL.md").read_text(encoding="utf-8") == CONTENT
    assert not any("skill-backups" in p for p in _file_tree(tmp_path))
    original_rename = Path.rename

    def fail_commit(path, target):
        if path.name.startswith(".import-"):
            raise OSError("commit failed")
        return original_rename(path, target)

    with patch.object(Path, "rename", fail_commit), pytest.raises(OSError):
        skills.install_package(
            str(tmp_path), package(), replace=True, expected_version=result["version"]
        )
    assert skills.detail(str(tmp_path), "sales")["version"] == result["version"]
    deleted = skills.remove(str(tmp_path), "sales", result["version"])
    assert Path(deleted["backup"]).is_dir()
    assert skills.listing(str(tmp_path))["items"] == []


def test_invalid_skill_diagnostics(tmp_path):
    folder = tmp_path / ".agents/skills/broken"
    folder.mkdir(parents=True)
    (folder / "SKILL.md").write_text("bad", encoding="utf-8")
    result = skills.listing(str(tmp_path))
    assert result["items"] == []
    assert result["diagnostics"][0]["name"] == "broken"


@pytest.mark.skipif(os.name != "nt", reason="Windows protected DACL regression")
def test_windows_import_and_legacy_repair_inherit_parent_acl(tmp_path):
    import ctypes
    from ctypes import wintypes

    advapi = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi.GetFileSecurityW.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetFileSecurityW.restype = wintypes.BOOL
    advapi.GetSecurityDescriptorControl.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(wintypes.WORD), ctypes.POINTER(wintypes.DWORD),
    ]
    advapi.GetSecurityDescriptorControl.restype = wintypes.BOOL

    def protected(path):
        needed = wintypes.DWORD()
        advapi.GetFileSecurityW(str(path), 4, None, 0, ctypes.byref(needed))
        assert needed.value > 0
        descriptor = ctypes.create_string_buffer(needed.value)
        assert advapi.GetFileSecurityW(str(path), 4, descriptor, needed.value, ctypes.byref(needed))
        control, revision = wintypes.WORD(), wintypes.DWORD()
        assert advapi.GetSecurityDescriptorControl(
            descriptor, ctypes.byref(control), ctypes.byref(revision)
        )
        return bool(control.value & 0x1000)

    first = skills.install_package(str(tmp_path), package())
    folder = Path(first["base_directory"])
    assert not protected(folder)
    legacy = Path(tempfile.mkdtemp(prefix=".legacy-", dir=folder.parent))
    for name, data in package().items():
        destination = legacy / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    skills.remove(str(tmp_path), "sales", first["version"])
    legacy.rename(folder)
    was_protected = protected(folder)  # Python >=3.13 creates a protected DACL.
    fixed = skills.repair_permissions(str(tmp_path), "sales", first["version"])
    assert not protected(folder)
    assert fixed["version"] == first["version"]
    assert protected(Path(fixed["backup"])) == was_protected
    assert skills._files(folder) == package()


def test_repair_permissions_preserves_content_backup_and_conflict(tmp_path):
    first = skills.install_package(str(tmp_path), package())
    with pytest.raises(skills.SkillConflictError):
        skills.repair_permissions(str(tmp_path), "sales", "0" * 64)
    assert not (tmp_path / ".agents/skill-backups").exists()
    fixed = skills.repair_permissions(str(tmp_path), "sales", first["version"])
    assert skills._files(Path(fixed["base_directory"])) == package()
    assert skills._files(Path(fixed["backup"])) == package()
    assert fixed["version"] == first["version"]


def test_explicit_disabled_skill_stops_before_model_or_commands(tmp_path):
    skills.install_package(str(tmp_path), package())
    state = session(tmp_path)
    with patch("veripatch.studio_executor.SafeStudioCommandRunner.run") as run:
        result = asyncio.run(
            StudioAgent(None, StudioStore(tmp_path / "state.db")).handle(
                state, "$sales 分析示例，实际执行脚本"
            )
        )
    run.assert_not_called()
    assert result.status == "paused"
    assert "保存使用方式" in result.messages[-1].content
    assert state.active_skill_contents == {}
    assert any(message.role == "user" and "$sales" in message.content for message in state.messages)


def test_automatic_loading_is_progressive_and_persists(tmp_path):
    content = CONTENT + "\n" + "完整指导\n" * 300  # More than default read's 240 lines.
    skills.install_package(str(tmp_path), {**package(), "SKILL.md": content.encode()})
    state = session(tmp_path, skill_modes={"sales": "auto"})
    store = StudioStore(tmp_path / "state.db")

    class Model:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                assert context["skills"]["selected"] == []
                available = context["skills"]["available"]
                assert available[0]["name"] == "sales"
                assert "content" not in available[0]
                return StudioReply(
                    decision=StudioDecision(
                        action=StudioAction.READ,
                        rationale="按需使用销售指导",
                        path=available[0]["location"],
                    )
                )
            loaded = context["skills"]["selected"][0]
            assert loaded["content"] == content
            assert "scripts/analyze.py" in [f["path"] for f in loaded["files"]]
            assert all("content" not in f for f in loaded["files"])
            assert context["skills"]["selected"] == [
                store.load(state.session_id).active_skill_contents["sales"]
            ]
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="说明已加载",
                    message="已读取销售统计指导，脚本尚未执行。",
                )
            )

    asyncio.run(StudioAgent(Model(), store).handle(state, "如何统计销售报表？"))
    assert state.status == "idle"
    assert state.messages[-1].content == "已读取销售统计指导，脚本尚未执行。"
    assert any(o.kind == "skill_loaded" for o in state.observations)
    assert state.changed_files == []
    assert len(state.active_skill_contents) == 1


def test_explicit_invocation_and_disabled(tmp_path):
    skills.install(str(tmp_path), CONTENT)
    assert skills.explicit_names("用 $sales 分析", {"sales"}) == ["sales"]
    assert skills.explicit_names("示例 `$sales` 或 \\$sales", {"sales"}) == []
    state = session(tmp_path, skill_modes={"sales": "disabled"})
    assert skills.catalog(state) == []
    with pytest.raises(ValueError, match="禁用"):
        skills.activate(state, "sales")
    state.skill_modes["sales"] = "auto"

    class Model:
        async def decide(self, context):
            assert context["skills"]["selected"][0]["name"] == "sales"
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RESPOND,
                    rationale="已按用户指定加载",
                    message="销售指导已就绪。",
                )
            )

    asyncio.run(
        StudioAgent(Model(), StudioStore(tmp_path / "state.db")).handle(
            state, "$sales 介绍统计口径"
        )
    )
    assert state.status == "idle"
    assert state.messages[-1].content == "销售指导已就绪。"


def test_loaded_skill_does_not_authorize_its_script(tmp_path):
    skills.install_package(str(tmp_path), package())
    state = session(tmp_path, skill_modes={"sales": "auto"}, permission_mode="ask")

    class Model:
        calls = 0

        async def decide(self, context):
            self.calls += 1
            if self.calls == 1:
                return StudioReply(
                    decision=StudioDecision(
                        action=StudioAction.READ,
                        rationale="加载销售指导",
                        path=context["skills"]["available"][0]["location"],
                    )
                )
            return StudioReply(
                decision=StudioDecision(
                    action=StudioAction.RUN_COMMAND,
                    rationale="统计销售数据",
                    command=["python", str(tmp_path / ".agents/skills/sales/scripts/analyze.py")],
                )
            )

    with patch("veripatch.studio_executor.SafeStudioCommandRunner.run") as run:
        asyncio.run(
            StudioAgent(Model(), StudioStore(tmp_path / "state.db")).handle(
                state,
                "统计这份销售报表",
            )
        )
    run.assert_not_called()
    assert state.status == "waiting_permission"
    assert state.pending_permission is not None
    assert state.pending_permission.decision["action"] == StudioAction.RUN_COMMAND.value
    assert state.action_grants == []


def test_compaction_all_branches_keep_active_skills(tmp_path):
    agent = StudioAgent(None, StudioStore(tmp_path / "state.db"))
    context = {"skills": {"selected": [{"content": "不可截断" * 2000}]}}
    assert agent._fit_context(context)[0]["skills"] == context["skills"]
    assert agent._compact_retry_context(context)["skills"] == context["skills"]
    assert agent._recovery_context(session(tmp_path), context)["skills"] == context["skills"]


def test_reject_linked_skill_resources(tmp_path):
    skills.install_package(str(tmp_path), package())
    root = tmp_path / ".agents/skills/sales"
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("do not read", encoding="utf-8")
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Directory symlinks require Windows Developer Mode or privilege")
    with pytest.raises(ValueError, match="链接"):
        skills.detail(str(tmp_path), "sales")
    assert secret.read_text(encoding="utf-8") == "do not read"


def test_streaming_body_limit(tmp_path):
    app = FastAPI()
    app.include_router(create_studio_router(Settings(database_path=tmp_path / "state.db")))
    with TestClient(app) as client:

        def chunks():
            for _ in range(17):
                yield b" " * (1024 * 1024)

        response = client.post(
            "/studio-api/sessions/anything/skills/preview",
            content=chunks(),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413


def test_api_package_preview_management_and_modes(tmp_path):
    db = tmp_path / "state.db"
    store = StudioStore(db)
    state = session(tmp_path)
    store.save(state, "created", {})
    app = FastAPI()
    app.include_router(create_studio_router(Settings(database_path=db)))
    url = "/studio-api/sessions/skills/skills"
    with TestClient(app) as client:
        body = {"archive": archive({f"sales/{k}": v for k, v in package().items()})}
        result = client.post(url + "/preview", json=body)
        assert result.status_code == 200
        assert not (tmp_path / ".agents").exists()
        installed = client.post(url, json=body)
        assert installed.status_code == 201
        listing = client.get(url).json()
        assert listing["modes"] == {"sales": "auto"}
        assert "content" not in listing["items"][0]
        assert (
            client.get(url + "/sales/resource", params={"path": "../state.db"}).status_code == 400
        )
        assert (
            client.get(url + "/sales/resource", params={"path": "references/rules.md"}).json()[
                "kind"
            ]
            == "text"
        )
        assert client.put(url, json={"modes": {"sales": "pinned"}}).status_code == 200
        assert store.load("skills").enabled_skills == ["sales"]
        assert client.put(url, json={"modes": {"absent": "auto"}}).status_code == 400
        info = client.get(url + "/sales").json()
        repaired = client.post(url + "/sales/repair", params={"version": info["version"]})
        assert repaired.status_code == 200
        assert repaired.json()["version"] == info["version"]
        assert Path(repaired.json()["backup"]).is_dir()
        assert store.load("skills").enabled_skills == ["sales"]
        assert client.post(url + "/sales/repair", params={"version": "0" * 64}).status_code == 409
        edited = client.patch(
            url + "/sales",
            json={"content": CONTENT + "\n补充。", "expected_version": info["version"]},
        )
        assert edited.status_code == 200
        assert len(edited.json()["files"]) == 4
        assert (
            client.patch(
                url + "/sales", json={"content": CONTENT, "expected_version": info["version"]}
            ).status_code
            == 409
        )
        assert client.delete(url + "/sales", params={"version": info["version"]}).status_code == 409
        state = store.load("skills")
        state.status = "running"
        store.save(state, "running", {})
        assert (
            client.post(url + "/sales/repair", params={"version": info["version"]}).status_code
            == 409
        )
        assert client.post(url, json=body).status_code == 409
        assert (
            client.patch(
                url + "/sales",
                json={"content": CONTENT, "expected_version": edited.json()["version"]},
            ).status_code
            == 409
        )
        assert (
            client.delete(url + "/sales", params={"version": edited.json()["version"]}).status_code
            == 409
        )
        state.status = "completed"
        store.save(state, "done", {})
        removed = client.delete(url + "/sales", params={"version": edited.json()["version"]})
        assert removed.status_code == 200
        assert Path(removed.json()["backup"]).is_dir()
        assert store.load("skills").enabled_skills == []
        assert client.get(url).json()["items"] == []
