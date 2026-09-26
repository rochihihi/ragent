import json

from veripatch.verification_discovery import discover_verification


def test_manual_command_wins(tmp_path):
    assert discover_verification(tmp_path, [], ["custom", "test"])[0]["source"] == "manual"


def test_shared_python_change_uses_full_suite(tmp_path):
    (tmp_path / "conftest.py").write_text("pass")
    (tmp_path / "test_conftest.py").write_text("def test_ok(): pass")
    assert discover_verification(tmp_path, ["conftest.py"], [])[0]["command"] == [
        "python",
        "-m",
        "pytest",
        "-q",
    ]


def test_verification_labels_and_typecheck():
    from veripatch.studio_agent import StudioAgent

    assert "测试" not in StudioAgent._verification_label(["python", "-m", "py_compile", "a.py"])
    assert "完整" not in StudioAgent._verification_label(
        ["python", "-m", "pytest", "-q", "test_a.py"]
    )
    assert StudioAgent.is_verification_command(["npm", "--prefix", "frontend", "run", "typecheck"])


def test_python_config_and_syntax_are_distinct(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (tmp_path / "hello.py").write_text("pass")
    (tmp_path / "test_other.py").write_text("def test_ok(): pass")
    candidates = discover_verification(tmp_path, ["hello.py"], [])
    assert candidates[0]["command"] == ["python", "-m", "pytest", "-q"]
    assert candidates[1]["kind"] == "syntax"


def test_javascript_lockfile_and_placeholder(tmp_path):
    (tmp_path / "package.json").write_text(
        json.dumps(
            {"scripts": {"test": "echo no test specified && exit 1", "build": "vite build"}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "pnpm-lock.yaml").write_text("", encoding="utf-8")
    result = discover_verification(tmp_path, ["index.js"], [])
    assert result[0]["command"] == ["pnpm", "run", "build"]


def test_malformed_metadata_and_empty_project(tmp_path):
    (tmp_path / "package.json").write_text("broken", encoding="utf-8")
    assert discover_verification(tmp_path, [], []) == []


def test_rust_and_go_candidates(tmp_path):
    (tmp_path / "Cargo.toml").write_text("", encoding="utf-8")
    (tmp_path / "go.mod").write_text("module example", encoding="utf-8")
    assert discover_verification(tmp_path, ["main.rs"], [])[0]["command"] == ["cargo", "test"]
    assert discover_verification(tmp_path, ["main.go"], [])[0]["command"] == ["go", "test", "./..."]


def test_associated_tests_without_config(tmp_path):
    (tmp_path / "hello.py").write_text("pass")
    (tmp_path / "test_hello.py").write_text("def test_ok(): pass")
    result = discover_verification(tmp_path, ["hello.py"], [])
    assert result[0]["command"] == ["python", "-m", "pytest", "-q", "test_hello.py"]


def test_dependency_graph_selects_test_through_import_chain(tmp_path):
    (tmp_path / "core.py").write_text("def normalize(v): return v.strip()\n")
    (tmp_path / "service.py").write_text(
        "from core import normalize\n\ndef process(v): return normalize(v)\n"
    )
    (tmp_path / "test_api.py").write_text(
        "from service import process\n\ndef test_process(): assert process(' x ') == 'x'\n"
    )

    result = discover_verification(tmp_path, ["core.py"], [])

    assert result[0]["command"] == ["python", "-m", "pytest", "-q", "test_api.py"]
    assert result[0]["source"] == "dependency-related test files"


def test_empty_pytest_config_uses_syntax(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]")
    (tmp_path / "hello.py").write_text("pass")
    assert discover_verification(tmp_path, ["hello.py"], [])[0]["kind"] == "syntax"
    assert discover_verification(tmp_path, ["deleted.py"], []) == []


def test_nested_frontend_check(tmp_path):
    frontend = tmp_path / "frontend"
    frontend.mkdir()
    (frontend / "package.json").write_text(json.dumps({"scripts": {"check": "tsc --noEmit"}}))
    assert discover_verification(tmp_path, ["frontend/app.ts"], [])[0]["command"] == [
        "npm",
        "--prefix",
        "frontend",
        "run",
        "check",
    ]
