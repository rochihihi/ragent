"""Python AST index for symbols, dependencies, calls, and impact analysis."""

from __future__ import annotations

import ast
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRECTORIES = {
    ".git", ".hg", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
    ".venv", "__pycache__", "build", "dist", "node_modules", "venv",
}


@dataclass(frozen=True, slots=True)
class SymbolRecord:
    name: str
    qualified_name: str
    kind: str
    path: str
    line: int
    end_line: int
    docstring: str | None = None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "name": self.name, "qualified_name": self.qualified_name, "kind": self.kind,
            "path": self.path, "line": self.line, "end_line": self.end_line,
            "docstring": self.docstring,
        }


@dataclass(frozen=True, slots=True)
class ImportRecord:
    path: str
    module: str
    name: str | None
    alias: str
    line: int
    target_path: str | None
    target_symbol: str | None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "path": self.path, "module": self.module, "name": self.name,
            "alias": self.alias, "line": self.line, "target_path": self.target_path,
            "target_symbol": self.target_symbol,
        }


@dataclass(frozen=True, slots=True)
class CallRecord:
    path: str
    caller: str
    callee: str
    line: int
    target_path: str | None
    target_symbol: str | None

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "path": self.path, "caller": self.caller, "callee": self.callee,
            "line": self.line, "target_path": self.target_path,
            "target_symbol": self.target_symbol,
        }


@dataclass(frozen=True, slots=True)
class _ImportFact:
    module: str
    name: str | None
    alias: str
    line: int
    level: int


@dataclass(frozen=True, slots=True)
class _CallFact:
    caller: str
    callee: str
    line: int


def _expression_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _expression_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return None


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self, relative_path: str) -> None:
        self.relative_path = relative_path
        self.scope: list[str] = []
        self.records: list[SymbolRecord] = []
        self.imports: list[_ImportFact] = []
        self.calls: list[_CallFact] = []

    def _record(self, node: ast.AST, name: str, kind: str) -> None:
        docstring = (
            ast.get_docstring(node)
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            else None
        )
        self.records.append(
            SymbolRecord(
                name=name,
                qualified_name=".".join([*self.scope, name]),
                kind=kind,
                path=self.relative_path,
                line=getattr(node, "lineno", 1),
                end_line=getattr(node, "end_lineno", getattr(node, "lineno", 1)),
                docstring=docstring,
            )
        )

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            self.imports.append(
                _ImportFact(item.name, None, item.asname or item.name.split(".")[0], node.lineno, 0)
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for item in node.names:
            if item.name != "*":
                self.imports.append(
                    _ImportFact(
                        node.module or "", item.name, item.asname or item.name,
                        node.lineno, node.level,
                    )
                )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._record(node, node.name, "class")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._record(node, node.name, "function")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._record(node, node.name, "async_function")
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_Call(self, node: ast.Call) -> None:
        callee = _expression_name(node.func)
        if callee:
            self.calls.append(
                _CallFact(".".join(self.scope) or "<module>", callee, node.lineno)
            )
        self.generic_visit(node)


class PythonSymbolIndex:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.records: list[SymbolRecord] = []
        self.imports: list[ImportRecord] = []
        self.calls: list[CallRecord] = []
        self.parse_errors: dict[str, str] = {}
        self._modules: dict[str, str] = {}
        self._path_modules: dict[str, str] = {}
        self._outgoing: dict[str, set[str]] = defaultdict(set)
        self._incoming: dict[str, set[str]] = defaultdict(set)
        self._visitors: dict[str, _SymbolVisitor] = {}

    @staticmethod
    def _module_name(relative: str) -> str:
        parts = list(Path(relative).with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts)

    def _absolute_module(self, path: str, module: str, level: int) -> str:
        if not level:
            return module
        current = self._path_modules[path].split(".") if self._path_modules[path] else []
        if Path(path).name != "__init__.py":
            current = current[:-1]
        remove = max(0, level - 1)
        if remove:
            current = current[:-remove] if remove <= len(current) else []
        return ".".join([*current, *[part for part in module.split(".") if part]])

    def _target_module(self, module: str) -> tuple[str | None, str]:
        candidate = module
        while candidate:
            if candidate in self._modules:
                return self._modules[candidate], candidate
            candidate = candidate.rpartition(".")[0]
        return None, module

    def build(self) -> PythonSymbolIndex:
        self.parse_errors.clear()
        self._visitors.clear()
        for path in self.root.rglob("*.py"):
            if any(part in IGNORED_DIRECTORIES for part in path.relative_to(self.root).parts):
                continue
            self._parse(path.relative_to(self.root).as_posix())
        self._resolve()
        return self

    def refresh(self, paths: list[str]) -> PythonSymbolIndex:
        """Reparse changed files and deterministically rebuild relationship edges."""
        for item in paths:
            relative = item.replace("\\", "/")
            self._visitors.pop(relative, None)
            self.parse_errors.pop(relative, None)
            target = (self.root / relative).resolve()
            if (
                target.is_file()
                and target.is_relative_to(self.root)
                and target.suffix == ".py"
                and not any(part in IGNORED_DIRECTORIES for part in Path(relative).parts)
            ):
                self._parse(relative)
        self._resolve()
        return self

    def _parse(self, relative: str) -> None:
        try:
            tree = ast.parse((self.root / relative).read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            self.parse_errors[relative] = str(exc)
            return
        visitor = _SymbolVisitor(relative)
        visitor.visit(tree)
        self._visitors[relative] = visitor

    def _resolve(self) -> None:
        self.records = [record for visitor in self._visitors.values() for record in visitor.records]
        self.imports.clear()
        self.calls.clear()
        self._modules.clear()
        self._path_modules = {
            path: self._module_name(path) for path in self._visitors
        }
        self._modules = {
            module: path for path, module in self._path_modules.items() if module
        }
        self._outgoing.clear()
        self._incoming.clear()

        symbols = {
            f"{self._path_modules[record.path]}.{record.qualified_name}".strip("."): record
            for record in self.records
        }
        aliases: dict[str, dict[str, tuple[str, str | None]]] = defaultdict(dict)
        for path, visitor in self._visitors.items():
            for import_fact in visitor.imports:
                module = self._absolute_module(path, import_fact.module, import_fact.level)
                imported = (
                    f"{module}.{import_fact.name}".strip(".")
                    if import_fact.name
                    else module
                )
                target_path, _ = self._target_module(imported)
                exact_module = imported in self._modules
                if exact_module:
                    target_symbol = None
                    alias_target = (
                        import_fact.alias
                        if import_fact.alias == imported.split(".")[0]
                        else imported
                    )
                else:
                    target_path, _ = self._target_module(module)
                    target_symbol = imported if import_fact.name and target_path else None
                    alias_target = target_symbol or imported
                import_record = ImportRecord(
                    path,
                    module,
                    import_fact.name,
                    import_fact.alias,
                    import_fact.line,
                    target_path,
                    target_symbol,
                )
                self.imports.append(import_record)
                aliases[path][import_fact.alias] = (alias_target, target_path)
                if target_path and target_path != path:
                    self._link(path, target_path)

        for path, visitor in self._visitors.items():
            module = self._path_modules[path]
            for call_fact in visitor.calls:
                root, dot, rest = call_fact.callee.partition(".")
                call_target_symbol: str | None = None
                call_target_path: str | None = None
                imported_alias = aliases[path].get(root)
                if imported_alias:
                    base, call_target_path = imported_alias
                    call_target_symbol = f"{base}.{rest}" if dot else base
                elif call_fact.callee.startswith("self.") and "." in call_fact.caller:
                    owner = call_fact.caller.rsplit(".", 1)[0]
                    call_target_symbol = (
                        f"{module}.{owner}.{call_fact.callee[5:]}".strip(".")
                    )
                else:
                    candidate = f"{module}.{call_fact.callee}".strip(".")
                    if candidate in symbols:
                        call_target_symbol = candidate
                        call_target_path = symbols[candidate].path
                if call_target_symbol in symbols:
                    call_target_path = symbols[call_target_symbol].path
                call_record = CallRecord(
                    path,
                    f"{module}.{call_fact.caller}".strip("."),
                    call_fact.callee,
                    call_fact.line,
                    call_target_path,
                    call_target_symbol,
                )
                self.calls.append(call_record)
                if call_target_path and call_target_path != path:
                    self._link(path, call_target_path)

    def _link(self, source: str, target: str) -> None:
        self._outgoing[source].add(target)
        self._incoming[target].add(source)

    def lookup(self, query: str, limit: int = 20) -> list[SymbolRecord]:
        tokens = {token.casefold() for token in query.replace(".", " ").split() if token}
        if not tokens:
            return []

        def score(record: SymbolRecord) -> tuple[int, int, str]:
            name = record.name.casefold()
            qualified = record.qualified_name.casefold()
            path = record.path.casefold()
            doc = (record.docstring or "").casefold()
            points = 0
            for token in tokens:
                if token == name:
                    points += 10
                elif token in name:
                    points += 6
                if token in qualified:
                    points += 4
                if token in path:
                    points += 2
                if token in doc:
                    points += 1
            return (-points, record.line, record.path)

        matches = [record for record in self.records if score(record)[0] < 0]
        return sorted(matches, key=score)[:limit]

    @staticmethod
    def _walk(graph: dict[str, set[str]], seeds: set[str], depth: int) -> list[str]:
        distance = {path: 0 for path in seeds}
        queue = deque(seeds)
        while queue:
            current = queue.popleft()
            if distance[current] >= depth:
                continue
            for target in sorted(graph.get(current, set())):
                if target not in distance:
                    distance[target] = distance[current] + 1
                    queue.append(target)
        ordered = sorted(distance.items(), key=lambda item: (item[1], item[0]))
        return [path for path, _ in ordered if path not in seeds]

    def dependencies(self, paths: list[str], depth: int = 2) -> list[str]:
        return self._walk(self._outgoing, {p.replace("\\", "/") for p in paths}, depth)

    def dependents(self, paths: list[str], depth: int = 2) -> list[str]:
        return self._walk(self._incoming, {p.replace("\\", "/") for p in paths}, depth)

    def related_files(self, paths: list[str], depth: int = 2, limit: int = 40) -> list[str]:
        seeds = {p.replace("\\", "/") for p in paths}
        related = [*self.dependencies(list(seeds), depth), *self.dependents(list(seeds), depth)]
        return list(dict.fromkeys(related))[:limit]

    def indexed_paths(self) -> list[str]:
        return sorted(self._path_modules)

    def related_tests(self, paths: list[str], depth: int = 8) -> list[str]:
        seeds = {p.replace("\\", "/") for p in paths}
        tests = {
            path for path in self._path_modules
            if Path(path).name.startswith("test_") or Path(path).name.endswith("_test.py")
        }
        return [
            test for test in sorted(tests)
            if test in seeds or seeds.intersection(self.dependencies([test], depth))
        ]

    def impact(self, paths: list[str], depth: int = 3) -> dict[str, list[str]]:
        normalized = list(dict.fromkeys(path.replace("\\", "/") for path in paths))
        return {
            "changed": normalized,
            "dependencies": self.dependencies(normalized, depth),
            "dependents": self.dependents(normalized, depth),
            "tests": self.related_tests(normalized),
        }

    def relationships(self, paths: list[str]) -> dict[str, list[dict[str, str | int | None]]]:
        selected = {path.replace("\\", "/") for path in paths}
        return {
            "imports": [
                item.as_dict() for item in self.imports
                if item.path in selected or item.target_path in selected
            ][:80],
            "calls": [
                item.as_dict() for item in self.calls
                if item.path in selected or item.target_path in selected
            ][:80],
        }

    def summary(self) -> dict[str, int]:
        return {
            "python_files": len(self._path_modules),
            "symbols": len(self.records),
            "imports": len(self.imports),
            "calls": len(self.calls),
            "parse_errors": len(self.parse_errors),
        }
