from pathlib import Path

from veripatch.indexing import PythonSymbolIndex


def test_symbol_index_finds_function(tmp_path: Path) -> None:
    (tmp_path / "sample.py").write_text(
        'def calculate_total(value: int) -> int:\n    """Calculate a total."""\n    return value\n',
        encoding="utf-8",
    )
    index = PythonSymbolIndex(tmp_path).build()
    matches = index.lookup("calculate total")
    assert matches
    assert matches[0].name == "calculate_total"
    assert index.summary()["symbols"] == 1


def test_symbol_index_handles_nested_async_symbols_and_parse_errors(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text(
        'class Service:\n    """API service."""\n    async def fetch(self):\n        return 1\n',
        encoding="utf-8",
    )
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    ignored = tmp_path / ".venv"
    ignored.mkdir()
    (ignored / "hidden.py").write_text("def hidden(): pass\n", encoding="utf-8")
    index = PythonSymbolIndex(tmp_path).build()
    matches = index.lookup("Service fetch")
    assert [record.qualified_name for record in matches] == ["Service.fetch", "Service"]
    assert matches[1].as_dict()["docstring"] == "API service."
    assert "broken.py" in index.parse_errors
    assert index.lookup("  ") == []
    assert index.summary() == {
        "python_files": 1,
        "symbols": 2,
        "imports": 0,
        "calls": 0,
        "parse_errors": 1,
    }


def test_index_resolves_cross_file_calls_and_impacted_tests(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    tests = tmp_path / "tests"
    package.mkdir()
    tests.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text(
        "def normalize(value):\n    return value.strip()\n", encoding="utf-8"
    )
    (package / "service.py").write_text(
        "from .core import normalize\n\ndef process(value):\n    return normalize(value)\n",
        encoding="utf-8",
    )
    (tests / "test_service.py").write_text(
        "from pkg.service import process\n\n"
        "def test_process():\n    assert process(' x ') == 'x'\n",
        encoding="utf-8",
    )

    index = PythonSymbolIndex(tmp_path).build()

    assert index.dependencies(["tests/test_service.py"], depth=3) == [
        "pkg/service.py",
        "pkg/core.py",
    ]
    assert index.dependents(["pkg/core.py"], depth=3) == [
        "pkg/service.py",
        "tests/test_service.py",
    ]
    assert index.related_tests(["pkg/core.py"]) == ["tests/test_service.py"]
    assert index.impact(["pkg/core.py"])["tests"] == ["tests/test_service.py"]
    call = next(item for item in index.calls if item.callee == "normalize")
    assert call.target_symbol == "pkg.core.normalize"
    assert call.target_path == "pkg/core.py"

    (package / "core.py").write_text(
        "def clean(value):\n    return value.strip()\n", encoding="utf-8"
    )
    index.refresh(["pkg/core.py"])
    assert index.lookup("clean")[0].path == "pkg/core.py"
    assert not index.lookup("normalize")


def test_index_resolves_qualified_module_calls(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text("def load():\n    return 1\n", encoding="utf-8")
    (tmp_path / "consumer.py").write_text(
        "import pkg.core\n\ndef run():\n    return pkg.core.load()\n", encoding="utf-8"
    )

    index = PythonSymbolIndex(tmp_path).build()
    call = next(item for item in index.calls if item.callee == "pkg.core.load")

    assert call.target_symbol == "pkg.core.load"
    assert call.target_path == "pkg/core.py"
