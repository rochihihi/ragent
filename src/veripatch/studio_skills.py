"""Project-scoped instruction skills. Never execute imported content."""

import re
from pathlib import Path


def root_for(project: str) -> Path:
    project_path = Path(project).resolve()
    root = project_path / ".agents" / "skills"
    if not root.resolve().is_relative_to(project_path):
        raise ValueError("技能目录不能指向项目外部")
    return root


def parse(content: str) -> dict[str, str]:
    if len(content) > 12000:
        raise ValueError("单个技能不能超过 12000 字符")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n(.+)\Z", content.strip(), re.S)
    if not match:
        raise ValueError("需要包含 name 和 description 的 YAML 头部及正文")
    fields = {}
    for key in ("name", "description"):
        item = re.search(rf"^{key}:\s*(.+)$", match[1], re.M)
        if not item:
            raise ValueError(f"缺少 {key}")
        fields[key] = item[1].strip().strip("\"'")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", fields["name"]):
        raise ValueError("name 只能包含小写字母、数字、下划线和短横线")
    if not fields["description"] or fields["description"] in {"|", ">"}:
        raise ValueError("description 请使用单行文本")
    return {**fields, "content": content}


def discover(project: str) -> list[dict[str, str]]:
    root = root_for(project)
    result = []
    for path in sorted(root.glob("*/SKILL.md")):
        try:
            if not path.resolve().is_relative_to(root.resolve()) or path.stat().st_size > 48000:
                continue
            item = parse(path.read_text(encoding="utf-8-sig"))
            if item["name"] != path.parent.name:
                continue
            result.append(item)
        except (ValueError, OSError, UnicodeError):
            continue
    return result


def install(project: str, content: str) -> dict[str, str]:
    item = parse(content.lstrip("\ufeff"))
    root = root_for(project)
    target = root / item["name"] / "SKILL.md"
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("技能路径越界")
    target.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation: imported skills cannot silently replace existing files.
    with target.open("x", encoding="utf-8") as stream:
        stream.write(item["content"])
    return item


def selected(project: str, names: list[str]) -> list[dict[str, str]]:
    if not names:
        return []
    root = root_for(project)
    items = []
    for name in dict.fromkeys(names):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("无效技能名称")
        path = root / name / "SKILL.md"
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("技能路径越界")
        if not path.is_file() or path.stat().st_size > 48000:
            raise ValueError("选中的技能不存在或文件过大")
        item = parse(path.read_text(encoding="utf-8-sig"))
        if item["name"] != name:
            raise ValueError("技能名称与目录不匹配")
        items.append(item)
    if sum(len(item["content"]) for item in items) > 12000:
        raise ValueError("启用技能合计不能超过 12000 字符")
    return items
