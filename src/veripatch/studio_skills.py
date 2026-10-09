"""Bounded project skill packages and progressive disclosure; never execute imports."""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import os
import re
import shutil
import stat
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

import yaml

MAX_CONTENT = 12_000
MAX_FILES = 200
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_PACKAGE_BYTES = 8 * 1024 * 1024
MAX_ACTIVE_CHARS = 60_000
MODES = {"auto", "pinned", "disabled"}
_LOCK = threading.RLock()
_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")  # Legacy underscore names remain valid.
_RESERVED = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\..*)?\Z", re.I)


class SkillConflictError(ValueError):
    """A package changed after the user's preview."""


class _Loader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result:
                raise ValueError("YAML 字段必须为不重复的字符串键")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def root_for(project: str) -> Path:
    project_path = Path(project).resolve()
    root = project_path / ".agents" / "skills"
    if not root.resolve().is_relative_to(project_path):
        raise ValueError("技能目录不能指向项目外部")
    return root


def _name(name: str) -> str:
    if not _NAME.fullmatch(name) or _RESERVED.fullmatch(name):
        raise ValueError("无效技能名称：请使用小写字母、数字、短横线（兼容旧下划线名称）")
    return name


def parse(content: str) -> dict[str, Any]:
    content = content.lstrip("\ufeff")
    if len(content) > MAX_CONTENT:
        raise ValueError("单个技能不能超过 12000 字符")
    match = re.match(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n(.+)\Z", content.strip(), re.S)
    if not match:
        raise ValueError("需要包含 name 和 description 的 YAML 头部及正文")
    try:
        depth = 0
        for count, event in enumerate(yaml.parse(match[1], Loader=yaml.SafeLoader), 1):
            if isinstance(event, (yaml.MappingStartEvent, yaml.SequenceStartEvent)):
                depth += 1
            elif isinstance(event, (yaml.MappingEndEvent, yaml.SequenceEndEvent)):
                depth -= 1
            if count > 1000 or depth > 20 or isinstance(event, yaml.AliasEvent):
                raise ValueError("技能 YAML 过于复杂或含有别名")
        fields = yaml.load(match[1], Loader=_Loader)
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误：{exc}") from exc
    if not isinstance(fields, dict):
        raise ValueError("YAML 头部必须为字段映射")
    name, description = fields.get("name"), fields.get("description")
    if not isinstance(name, str):
        raise ValueError("缺少字符串 name")
    _name(name)
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("description 必须为 1–1024 字符的文本，支持 YAML 多行描述")
    compatibility = fields.get("compatibility", "")
    if not isinstance(compatibility, str) or len(compatibility) > 500:
        raise ValueError("compatibility 必须为最多 500 字符的文本")
    metadata = fields.get("metadata", {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError("metadata 必须为字符串键值映射")
    if not match[2].strip():
        raise ValueError("技能正文不能为空")
    return {
        "name": name,
        "description": description.strip(),
        "content": content,
        "compatibility": compatibility,
        "metadata": metadata,
    }


def _relative(value: str) -> str:
    value = value.replace("\\", "/")
    if (
        not value
        or len(value) > 240
        or any(
            part in {"", ".", ".."}
            or part.endswith((".", " "))
            or _RESERVED.fullmatch(part)
            or any(ord(c) < 32 or c in ':<>"|?*' for c in part)
            for part in value.split("/")
        )
    ):
        raise ValueError(f"不安全的技能文件路径：{value!r}")
    return PurePosixPath(value).as_posix()


def _linked(path: Path) -> bool:
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


def _directory(project: str, name: str) -> Path:
    root = root_for(project)
    path = root / _name(name)
    if _linked(path) or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("技能路径越界或包含目录链接")
    return path


def _files(path: Path) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    total = 0
    for directories, (current, dirs, names) in enumerate(os.walk(path, followlinks=False), 1):
        if directories > 1000:
            raise ValueError("技能包目录过多")
        for name in [*dirs, *names]:
            entry = Path(current) / name
            if _linked(entry) or not entry.resolve().is_relative_to(path.resolve()):
                raise ValueError("技能包不能包含链接或目录联接")
        for name in names:
            entry = Path(current) / name
            relative = _relative(entry.relative_to(path).as_posix())
            if not entry.is_file() or entry.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"技能文件不是普通文件或超过 2 MiB：{relative}")
            data = entry.read_bytes()
            total += len(data)
            if total > MAX_PACKAGE_BYTES or len(result) >= MAX_FILES:
                raise ValueError("技能包最多 200 个文件、合计 8 MiB")
            result[relative] = data
    return result


def _version(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for name, data in sorted(files.items()):
        digest.update(name.encode("utf-8") + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def detail(project: str, name: str) -> dict[str, Any]:
    path = _directory(project, name)
    if not path.is_dir():
        raise ValueError("选中的技能不存在")
    files = _files(path)
    try:
        item = parse(files["SKILL.md"].decode("utf-8-sig"))
    except (KeyError, UnicodeError) as exc:
        raise ValueError("技能缺少 UTF-8 的 SKILL.md") from exc
    if item["name"] != name:
        raise ValueError("技能名称与目录不匹配")
    return {
        **item,
        "location": str(path / "SKILL.md"),
        "base_directory": str(path),
        "files": [{"path": p, "size": len(data)} for p, data in sorted(files.items())],
        "version": _version(files),
    }


def listing(project: str) -> dict[str, Any]:
    root = root_for(project)
    items, diagnostics = [], []
    for path in sorted(root.iterdir()) if root.exists() else []:
        if path.name.startswith(".") or not path.is_dir():
            continue
        try:
            item = detail(project, path.name)
            items.append({k: v for k, v in item.items() if k != "content"})
        except (ValueError, OSError, UnicodeError) as exc:
            diagnostics.append({"name": path.name, "message": str(exc)})
    return {"items": items, "diagnostics": diagnostics}


def discover(project: str) -> list[dict[str, Any]]:
    """Compatibility API; management uses listing() and lazy detail()."""
    return [detail(project, item["name"]) for item in listing(project)["items"]]


def decode_package(
    *,
    content: str | None = None,
    files: list[dict[str, str]] | None = None,
    archive: str | None = None,
) -> dict[str, bytes]:
    if sum(value is not None for value in (content, files, archive)) != 1:
        raise ValueError("请选择一种导入来源：说明文本、文件夹或 ZIP")
    result: dict[str, bytes] = {}
    seen: set[str] = set()

    def add(name: str, data: bytes) -> None:
        name = _relative(name)
        if name.casefold() in seen:
            raise ValueError(f"技能包存在重复路径（含大小写冲突）：{name}")
        if len(data) > MAX_FILE_BYTES:
            raise ValueError(f"文件超过 2 MiB：{name}")
        seen.add(name.casefold())
        result[name] = data
        if len(result) > MAX_FILES or sum(map(len, result.values())) > MAX_PACKAGE_BYTES:
            raise ValueError("技能包最多 200 个文件、合计 8 MiB")

    try:
        if content is not None:
            add("SKILL.md", content.encode("utf-8"))
        elif files is not None:
            if len(files) > MAX_FILES:
                raise ValueError("技能包最多 200 个文件")
            for entry in files:
                add(entry["path"], base64.b64decode(entry["data"], validate=True))
        else:
            raw = base64.b64decode(archive or "", validate=True)
            if len(raw) > MAX_PACKAGE_BYTES:
                raise ValueError("ZIP 文件超过 8 MiB")
            with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
                entries = bundle.infolist()
                if len(entries) > MAX_FILES * 2:
                    raise ValueError("ZIP 条目过多")
                declared_total = 0
                for entry in entries:
                    original = entry.orig_filename
                    _relative(original.rstrip("/") if entry.is_dir() else original)
                    mode = entry.external_attr >> 16
                    if stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                        raise ValueError("ZIP 不能包含链接或特殊文件")
                    _relative(entry.filename.rstrip("/") if entry.is_dir() else entry.filename)
                    if entry.is_dir():
                        continue
                    declared_total += entry.file_size
                    if entry.file_size > MAX_FILE_BYTES or declared_total > MAX_PACKAGE_BYTES:
                        raise ValueError("ZIP 解压后体积超过限制")
                    with bundle.open(entry) as stream:
                        add(entry.filename, stream.read(MAX_FILE_BYTES + 1))
    except (binascii.Error, zipfile.BadZipFile, RuntimeError, NotImplementedError) as exc:
        raise ValueError("无法读取 ZIP 或 Base64 文件内容") from exc
    if "SKILL.md" not in result:
        candidates = [p for p in result if p.endswith("/SKILL.md") and p.count("/") == 1]
        if len(candidates) != 1:
            raise ValueError("请选择一个完整技能文件夹：根目录需要 SKILL.md")
        prefix = candidates[0].removesuffix("SKILL.md")
        if any(not p.startswith(prefix) for p in result):
            raise ValueError("技能包只能包含一个技能目录，不能夹带目录外文件")
        result = {p[len(prefix) :]: data for p, data in result.items()}
    try:
        parse(result["SKILL.md"].decode("utf-8-sig"))
    except UnicodeError as exc:
        raise ValueError("SKILL.md 必须使用 UTF-8 编码") from exc
    paths = {p.casefold() for p in result}
    spellings: dict[str, str] = {}
    for path in result:
        for parent in [PurePosixPath(path), *PurePosixPath(path).parents]:
            spelling = parent.as_posix()
            folded = spelling.casefold()
            if folded in spellings and spellings[folded] != spelling:
                raise ValueError("技能包存在大小写不一致的目录路径")
            spellings[folded] = spelling
    for name in paths:
        if any(
            parent.as_posix() in paths
            for parent in PurePosixPath(name).parents
            if parent.as_posix() != "."
        ):
            raise ValueError("技能包存在文件与目录冲突")
    return result


def preview(project: str, files: dict[str, bytes]) -> dict[str, Any]:
    item = parse(files["SKILL.md"].decode("utf-8-sig"))
    target = _directory(project, item["name"])
    existing = _version(_files(target)) if target.exists() else None
    return {
        **item,
        "files": [{"path": p, "size": len(data)} for p, data in sorted(files.items())],
        "existing_version": existing,
        "total_bytes": sum(map(len, files.values())),
        "warning": "导入不会运行脚本或安装依赖；请确认来源可信。",
    }


def _backup(project: str, target: Path) -> Path:
    root = Path(project).resolve() / ".agents" / "skill-backups"
    if _linked(root) or not root.resolve().is_relative_to(Path(project).resolve()):
        raise ValueError("技能备份目录不能指向项目外部")
    root.mkdir(parents=True, exist_ok=True)
    backup = root / f"{target.name}-{uuid4().hex}"
    target.rename(backup)
    return backup


def install_package(
    project: str,
    files: dict[str, bytes],
    *,
    replace: bool = False,
    expected_version: str | None = None,
) -> dict[str, Any]:
    with _LOCK:
        info = preview(project, files)
        root = root_for(project)
        target = _directory(project, info["name"])
        if target.exists():
            if not replace:
                raise FileExistsError("同名技能已存在")
            if not expected_version or info["existing_version"] != expected_version:
                raise SkillConflictError("技能在预览后已变更，请刷新并重新确认")
        elif replace:
            raise SkillConflictError("原技能已不存在，请重新预览导入")
        root.mkdir(parents=True, exist_ok=True)
        # Python 3.13's Windows mkdtemp uses a protected owner-only DACL.
        # Renaming it retains that DACL and blocks the sandbox account even
        # after the workspace has been granted access. Inherit the project's
        # permissions instead; do not grant new users or alter ancestor ACLs.
        staging = root / f".import-{uuid4().hex}"
        staging.mkdir()
        backup = None
        try:
            for relative, data in files.items():
                destination = staging / _relative(relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            if target.exists():
                backup = _backup(project, target)
            try:
                staging.rename(target)
            except BaseException:
                if backup is not None:
                    backup.rename(target)
                raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        return {**detail(project, info["name"]), "backup": str(backup) if backup else None}


def edit(project: str, name: str, content: str, expected_version: str) -> dict[str, Any]:
    if parse(content)["name"] != name:
        raise ValueError("编辑说明不能改名；请另行导入新技能")
    files = _files(_directory(project, name))
    files["SKILL.md"] = content.encode("utf-8")
    return install_package(project, files, replace=True, expected_version=expected_version)


def install(project: str, content: str) -> dict[str, Any]:
    return install_package(project, decode_package(content=content))


def repair_permissions(project: str, name: str, expected_version: str) -> dict[str, Any]:
    """Reinstall unchanged bytes with inherited ACLs; preserve the original as a backup.

    This is an explicit management operation, not a sandbox bypass or an
    automatic rewrite of directories the user may have made private.
    """
    with _LOCK:
        path = _directory(project, name)
        info = detail(project, name)
        if info["version"] != expected_version:
            raise SkillConflictError("技能已变更，请刷新后再修复权限")
        return install_package(
            project, _files(path), replace=True, expected_version=expected_version
        )


def remove(project: str, name: str, expected_version: str) -> dict[str, str]:
    with _LOCK:
        target = _directory(project, name)
        if not target.is_dir() or _version(_files(target)) != expected_version:
            raise SkillConflictError("技能已变更或不存在，请刷新后重试")
        return {"name": name, "backup": str(_backup(project, target))}


def resource(project: str, name: str, relative: str) -> dict[str, Any]:
    files = _files(_directory(project, name))
    relative = _relative(relative)
    if relative not in files:
        raise ValueError("技能资源不存在")
    data = files[relative]
    try:
        text = data.decode("utf-8-sig")
        if "\x00" in text:
            raise UnicodeError("binary")
        return {
            "path": relative,
            "size": len(data),
            "kind": "text",
            "content": text[:32_000],
            "truncated": len(text) > 32_000,
        }
    except UnicodeError:
        return {"path": relative, "size": len(data), "kind": "binary", "content": ""}


def selected(project: str, names: list[str]) -> list[dict[str, Any]]:
    items = [detail(project, name) for name in dict.fromkeys(names)]
    if sum(len(item["content"]) for item in items) > MAX_ACTIVE_CHARS:
        raise ValueError("加载技能合计不能超过 60000 字符，请减少固定启用项")
    return items


def mode_for(session: Any, name: str) -> str:
    return session.skill_modes.get(name, "pinned" if name in session.enabled_skills else "disabled")


def catalog(session: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "description": item["description"],
            "location": item["location"],
            "compatibility": item["compatibility"],
        }
        for item in listing(session.repo_root)["items"]
        if mode_for(session, item["name"]) != "disabled"
    ]


def explicit_names(message: str, available: set[str]) -> list[str]:
    plain = re.sub(r"```.*?```|`[^`]*`", "", message, flags=re.S)
    names = re.findall(r"(?<![\\\w])\$([a-z0-9][a-z0-9_-]{0,63})(?![\w-])", plain)
    return list(dict.fromkeys(name for name in names if name in available))


def activate(session: Any, name: str) -> dict[str, Any]:
    if mode_for(session, name) == "disabled":
        raise ValueError("技能已禁用，请先在技能管理中启用")
    if name not in session.active_skill_contents:
        item = detail(session.repo_root, name)
        size = sum(len(i["content"]) for i in session.active_skill_contents.values())
        if size + len(item["content"]) > MAX_ACTIVE_CHARS:
            raise ValueError("本次加载的技能内容超过 60000 字符")
        session.active_skill_contents[name] = item
    return session.active_skill_contents[name]


def activation_for_read(session: Any, path: Path) -> dict[str, Any] | None:
    root = root_for(session.repo_root).resolve()
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        return None
    relative = resolved.relative_to(root)
    if len(relative.parts) != 2 or relative.parts[1] != "SKILL.md":
        return None
    return activate(session, relative.parts[0])
