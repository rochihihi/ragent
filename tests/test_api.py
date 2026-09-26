import asyncio
import shutil
import time
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from veripatch.api import _public_event, _test_provider_connection, create_app
from veripatch.config import Settings
from veripatch.quota import BalanceInfo, QuotaSnapshot


def test_health_and_missing_run(tmp_path: Path) -> None:
    app = create_app(Settings(database_path=tmp_path / "api.sqlite3"))
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/runs/missing")
    assert response.status_code == 404
    assert client.get("/runs/missing/result").status_code == 404
    assert client.get("/runs/missing/events").status_code == 404
    assert client.post("/runs/missing/resume").status_code == 404


def test_root_redirects_to_studio_and_provider_status_is_safe(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "ds-super-secret-value")
    client = TestClient(create_app(Settings(database_path=tmp_path / "api.sqlite3")))
    home = client.get("/")
    assert home.status_code == 200
    assert home.url.path == "/studio"
    assert '<div id="root"></div>' in home.text
    assert 'type="module"' in home.text
    providers = client.get("/providers").json()
    assert providers["deepseek"] == {
        "configured": True,
        "source": "environment",
        "model": "deepseek-v4-flash",
    }
    assert "secret" not in str(providers)
    assert client.get("/runs").json() == []


def test_provider_quota_endpoint_is_public_and_secret_free(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "veripatch.api.get_provider_quota",
        lambda provider, settings, **_kwargs: QuotaSnapshot(
            provider=provider,
            supported=True,
            configured=True,
            is_available=True,
            low_balance=False,
            low_balance_threshold="5",
            balances=[
                BalanceInfo(
                    currency="CNY",
                    total_balance="12.50",
                    granted_balance="2.50",
                    topped_up_balance="10.00",
                )
            ],
        ),
    )
    client = TestClient(create_app(Settings(database_path=tmp_path / "quota.sqlite3")))
    payload = client.get("/providers/deepseek/quota").json()
    assert payload["balances"][0]["total_balance"] == "12.50"
    assert payload["low_balance"] is False
    assert "api_key" not in str(payload).casefold()


def test_provider_credentials_can_be_configured_and_removed_without_echoing_secret(
    tmp_path: Path, monkeypatch
) -> None:
    configured = {"openai": False, "deepseek": False}
    saved: list[tuple[str, str]] = []

    def statuses():
        return {
            provider: {
                "configured": ready,
                "source": "keyring" if ready else "none",
            }
            for provider, ready in configured.items()
        }

    def save(provider: str, value: str) -> None:
        saved.append((provider, value))
        configured[provider] = True

    def remove(provider: str) -> bool:
        was_configured = configured[provider]
        configured[provider] = False
        return was_configured

    monkeypatch.setattr("veripatch.api.credential_status", statuses)
    monkeypatch.setattr("veripatch.api.save_api_key", save)
    monkeypatch.setattr("veripatch.api.delete_api_key", remove)
    monkeypatch.setattr(
        "veripatch.api.load_api_key",
        lambda provider: "ds-private-value" if configured[provider] else None,
    )
    client = TestClient(create_app(Settings(database_path=tmp_path / "credentials.sqlite3")))

    response = client.post("/providers/deepseek/credentials", json={"api_key": "ds-private-value"})
    assert response.status_code == 200
    assert response.json() == {
        "provider": "deepseek",
        "configured": True,
        "source": "keyring",
    }
    assert saved == [("deepseek", "ds-private-value")]
    assert "private-value" not in response.text

    reveal_url = "/providers/deepseek/credentials/reveal"
    assert client.get(reveal_url).status_code == 403
    response = client.get(reveal_url, headers={"x-veripatch-ui": "1"})
    assert response.json() == {"provider": "deepseek", "api_key": "ds-private-value"}
    assert response.headers["cache-control"] == "no-store"

    response = client.delete("/providers/deepseek/credentials")
    assert response.status_code == 200
    assert response.json()["removed"] is True
    assert response.json()["configured"] is False
    assert client.post("/providers/other/credentials", json={"api_key": "x"}).status_code == 404
    assert client.delete("/providers/other/credentials").status_code == 404
    assert (
        client.get(
            "/providers/other/credentials/reveal", headers={"x-veripatch-ui": "1"}
        ).status_code
        == 404
    )
    assert client.get(reveal_url, headers={"x-veripatch-ui": "1"}).status_code == 404
    assert client.post("/providers/openai/credentials", json={"api_key": ""}).status_code == 422
    oversized_secret = "sensitive-marker-" + ("x" * 10_000)
    response = client.post("/providers/openai/credentials", json={"api_key": oversized_secret})
    assert response.status_code == 422
    assert "sensitive-marker" not in response.text

    def fail_to_save(_provider: str, _value: str) -> None:
        raise RuntimeError("backend-sensitive-marker")

    monkeypatch.setattr("veripatch.api.save_api_key", fail_to_save)
    response = client.post("/providers/openai/credentials", json={"api_key": "valid-shape"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Credential store unavailable"
    assert "backend-sensitive-marker" not in response.text

    monkeypatch.setattr("veripatch.api.save_api_key", lambda _provider, _value: None)
    configured["openai"] = False
    response = client.post("/providers/openai/credentials", json={"api_key": "valid-shape"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Credential could not be persisted"


def test_environment_credentials_cannot_be_removed_from_gui(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "veripatch.api.credential_status",
        lambda: {
            "openai": {"configured": True, "source": "environment"},
            "deepseek": {"configured": False, "source": "none"},
        },
    )
    client = TestClient(create_app(Settings(database_path=tmp_path / "environment.sqlite3")))
    response = client.delete("/providers/openai/credentials")
    assert response.status_code == 409
    assert "environment variable" in response.json()["detail"]


def test_provider_model_selection_is_validated_and_persisted(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "models.sqlite3")
    client = TestClient(create_app(settings))
    models = client.get("/provider-models").json()
    assert models["deepseek"]["selected"] == "deepseek-v4-flash"
    assert models["deepseek"]["choices"] == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert models["openai"]["choices"] == [
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
    ]

    response = client.put("/providers/deepseek/model", json={"model": "deepseek-v4-pro"})
    assert response.json() == {"provider": "deepseek", "model": "deepseek-v4-pro"}
    assert client.put("/providers/deepseek/model", json={"model": "unsupported"}).status_code == 400
    assert client.put("/providers/other/model", json={"model": "x"}).status_code == 404
    assert client.put("/providers/openai/model", json={"model": "gpt-5.6-sol"}).status_code == 200

    restarted = TestClient(create_app(settings))
    assert restarted.get("/provider-models").json()["deepseek"]["selected"] == ("deepseek-v4-pro")


def test_openai_compatible_connection_is_configurable_and_persisted(tmp_path: Path) -> None:
    settings = Settings(database_path=tmp_path / "connections.sqlite3")
    client = TestClient(create_app(settings))

    response = client.put(
        "/providers/openai/connection",
        json={"base_url": "https://proxy.example/v1/", "model": "proxy-coding-model"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "provider": "openai",
        "base_url": "https://proxy.example/v1",
        "model": "proxy-coding-model",
    }
    restarted = TestClient(create_app(settings))
    assert restarted.get("/providers").json()["openai"]["base_url"] == ("https://proxy.example/v1")
    assert restarted.get("/provider-models").json()["openai"]["selected"] == ("proxy-coding-model")
    assert (
        client.put(
            "/providers/openai/connection",
            json={"base_url": "not-a-url", "model": "model"},
        ).status_code
        == 400
    )


def test_openai_models_can_be_discovered_with_typed_or_stored_key(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[tuple[str, str]] = []

    async def discover(base_url: str, api_key: str) -> list[str]:
        calls.append((base_url, api_key))
        return ["coding-large", "coding-small"]

    monkeypatch.setattr("veripatch.api._discover_openai_models", discover)
    monkeypatch.setattr("veripatch.api.load_api_key", lambda provider: "stored-key")
    client = TestClient(create_app(Settings(database_path=tmp_path / "discover.sqlite3")))

    typed = client.post(
        "/providers/openai/models/discover",
        json={"base_url": "https://proxy.example/v1/", "api_key": "typed-key"},
    )
    stored = client.post(
        "/providers/openai/models/discover",
        json={"base_url": "https://proxy.example/v1", "api_key": None},
    )

    assert typed.json() == {"models": ["coding-large", "coding-small"]}
    assert stored.status_code == 200
    assert calls == [
        ("https://proxy.example/v1", "typed-key"),
        ("https://proxy.example/v1", "stored-key"),
    ]
    discovered = client.get("/provider-models").json()["openai"]["choices"]
    assert "coding-large" in discovered
    restarted = TestClient(create_app(Settings(database_path=tmp_path / "discover.sqlite3")))
    assert "coding-small" in restarted.get("/provider-models").json()["openai"]["choices"]


def test_provider_connection_can_test_unsaved_proxy_settings_without_echoing_key(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[dict[str, str]] = []

    async def test_connection(**kwargs):
        calls.append(kwargs)
        return {
            "ok": True,
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "latency_ms": 321,
            "status_code": 200,
            "message": "连接成功，密钥与所选模型可以正常调用。",
        }

    monkeypatch.setattr("veripatch.api._test_provider_connection", test_connection)
    client = TestClient(create_app(Settings(database_path=tmp_path / "connection-test.sqlite3")))
    response = client.post(
        "/providers/openai/connection-test",
        json={
            "base_url": "https://proxy.example/v1/",
            "model": "coding-model",
            "api_key": "unsaved-private-key",
        },
    )

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert response.json()["latency_ms"] == 321
    assert "private-key" not in response.text
    assert calls == [
        {
            "provider": "openai",
            "base_url": "https://proxy.example/v1",
            "model": "coding-model",
            "api_key": "unsaved-private-key",
        }
    ]


def test_provider_connection_requires_a_key_and_rejects_unknown_provider(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr("veripatch.api.load_api_key", lambda _provider: None)
    client = TestClient(create_app(Settings(database_path=tmp_path / "connection-key.sqlite3")))
    body = {"model": "some-model", "base_url": "https://proxy.example/v1"}
    assert client.post("/providers/openai/connection-test", json=body).status_code == 400
    assert client.post("/providers/other/connection-test", json=body).status_code == 404


def test_official_openai_is_separate_from_proxy_configuration(tmp_path: Path, monkeypatch) -> None:
    keys = {"openai": "proxy-secret", "openai_official": "official-secret"}
    monkeypatch.setattr("veripatch.api.load_api_key", lambda provider: keys.get(provider))
    calls = []

    async def test_connection(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "provider": kwargs["provider"], "model": kwargs["model"],
                "latency_ms": 1, "status_code": 200, "message": "OK"}

    monkeypatch.setattr("veripatch.api._test_provider_connection", test_connection)
    client = TestClient(create_app(Settings(database_path=tmp_path / "official.sqlite3")))
    models = client.get("/provider-models").json()
    assert "gpt-6-astra" in models["openai_official"]["choices"]
    assert "gpt-6-sol" in models["openai_official"]["choices"]
    assert "gpt-6-luna" in models["openai_official"]["choices"]
    assert client.put(
        "/providers/openai/connection",
        json={"base_url": "https://proxy.example/v1", "model": "proxy-model"},
    ).status_code == 200
    response = client.post(
        "/providers/openai_official/connection-test",
        json={"model": "gpt-6-astra", "base_url": "https://proxy.example/v1"},
    )
    assert response.status_code == 200
    assert calls == [{"provider": "openai_official", "base_url": "https://api.openai.com/v1",
                      "model": "gpt-6-astra", "api_key": "official-secret"}]
    assert client.get("/providers").json()["openai"]["base_url"] == "https://proxy.example/v1"


def test_official_connection_test_uses_responses_endpoint(monkeypatch) -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={
            "output": [{"content": [{"type": "output_text", "text": "OK"}]}]
        })

    client_class = httpx.AsyncClient
    monkeypatch.setattr(
        "veripatch.api.httpx.AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler)),
    )
    result = asyncio.run(_test_provider_connection(
        provider="openai_official", base_url="https://api.openai.com/v1",
        model="gpt-6-astra", api_key="test-official-key",
    ))
    assert result["ok"] is True
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer test-official-key"


def test_web_directory_browser_lists_folders_and_handles_errors(tmp_path: Path) -> None:
    selected = tmp_path / "repository"
    child = selected / "child"
    child.mkdir(parents=True)
    client = TestClient(create_app(Settings(database_path=tmp_path / "picker.sqlite3")))
    assert client.get("/system/directories").status_code == 403
    headers = {"x-veripatch-ui": "1"}
    response = client.get("/system/directories", params={"path": str(selected)}, headers=headers)
    assert response.status_code == 200
    assert response.json()["current"] == str(selected.resolve())
    assert response.json()["directories"] == [{"name": "child", "path": str(child)}]
    assert (
        client.get(
            "/system/directories", params={"path": str(tmp_path / "missing")}, headers=headers
        ).status_code
        == 404
    )
    assert client.post("/system/select-directory", headers=headers).status_code == 404


def test_web_directory_browser_creates_a_child_folder(tmp_path: Path) -> None:
    parent = tmp_path / "workspace"
    parent.mkdir()
    client = TestClient(create_app(Settings(database_path=tmp_path / "picker.sqlite3")))
    headers = {"x-veripatch-ui": "1"}
    created = client.post(
        "/system/directories",
        headers=headers,
        json={"parent": str(parent), "name": "new-project"},
    )
    assert created.status_code == 201
    assert created.json()["path"] == str(parent / "new-project")
    assert (parent / "new-project").is_dir()
    duplicate = client.post(
        "/system/directories",
        headers=headers,
        json={"parent": str(parent), "name": "new-project"},
    )
    assert duplicate.status_code == 409
    escaped = client.post(
        "/system/directories",
        headers=headers,
        json={"parent": str(parent), "name": "../outside"},
    )
    assert escaped.status_code == 400


def test_web_directory_browser_has_desktop_shortcut(tmp_path: Path, monkeypatch) -> None:
    home = tmp_path / "user"
    desktop = home / "Desktop"
    desktop.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    client = TestClient(create_app(Settings(database_path=tmp_path / "desktop.sqlite3")))
    response = client.get("/system/directories", headers={"x-veripatch-ui": "1"})
    assert response.status_code == 200
    assert response.json()["directories"][0] == {
        "name": "桌面",
        "path": str(desktop.resolve()),
    }


def test_background_run_exposes_state_events_and_result(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    shutil.copytree(project_root / "examples" / "discount_bug", repository)
    settings = Settings(
        database_path=tmp_path / "api.sqlite3",
        test_timeout_seconds=30,
        test_runner="local",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post(
            "/runs",
            json={
                "repo_root": str(repository),
                "issue": {
                    "issue_id": "discount",
                    "title": "Incorrect discount",
                    "description": "A ten percent discount on 100 should equal 90.",
                },
                "provider": "scripted-demo",
                "runner": "local",
            },
        )
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        for _ in range(50):
            state_response = client.get(f"/runs/{run_id}")
            if state_response.json()["phase"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert state_response.json()["phase"] == "succeeded"
        public_state = state_response.json()
        for private in (
            "original_files",
            "working_file_hashes",
            "action_fingerprints",
            "pending_edit",
            "request_ids",
        ):
            assert private not in public_state
        events = client.get(f"/runs/{run_id}/events").json()
        assert any(event["event_type"] == "succeeded" for event in events)
        result = client.get(f"/runs/{run_id}/result")
        assert result.status_code == 200
        assert "percent / 100" in result.json()["diff"]
        assert client.get(f"/dashboard/{run_id}").status_code == 404
        assert client.get("/runs?limit=1&offset=0").json()[0]["run_id"] == run_id
        first_event = client.get(f"/runs/{run_id}/events?limit=1").json()[0]
        next_events = client.get(
            f"/runs/{run_id}/events?after={first_event['sequence']}&limit=2"
        ).json()
        assert all(event["sequence"] > first_event["sequence"] for event in next_events)
        assert client.post(f"/runs/{run_id}/resume").status_code == 409


def test_one_click_demo_uses_an_isolated_copy(tmp_path: Path) -> None:
    settings = Settings(
        database_path=tmp_path / "api.sqlite3",
        test_timeout_seconds=30,
        test_runner="local",
    )
    with TestClient(create_app(settings)) as client:
        response = client.post("/demo")
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        for _ in range(50):
            state = client.get(f"/runs/{run_id}").json()
            if state["phase"] in {"succeeded", "failed"}:
                break
            time.sleep(0.05)
        assert state["phase"] == "succeeded"
        workspace = Path(state["repo_root"])
        assert workspace.parent == tmp_path / "demo_workspaces"
        assert "percent / 100" in client.get(f"/runs/{run_id}/result").json()["diff"]


def test_create_run_rejects_invalid_repository_and_provider(tmp_path: Path) -> None:
    client = TestClient(create_app(Settings(database_path=tmp_path / "api.sqlite3")))
    payload = {
        "repo_root": str(tmp_path / "missing"),
        "issue": {"issue_id": "1", "title": "Bug", "description": "Broken"},
    }
    assert client.post("/runs", json=payload).status_code == 400
    payload["repo_root"] = str(tmp_path)
    payload["provider"] = "invalid"
    assert client.post("/runs", json=payload).status_code == 400

    payload["provider"] = "deepseek"
    payload["model"] = "unsupported"
    response = client.post("/runs", json=payload)
    assert response.status_code == 400
    assert "Unsupported model" in response.json()["detail"]

    payload["model"] = "deepseek-v4-flash"
    payload["reasoning_effort"] = "xhigh"
    response = client.post("/runs", json=payload)
    assert response.status_code == 400
    assert "Unsupported reasoning effort" in response.json()["detail"]


def test_public_event_drops_internal_fields_and_redacts_credentials() -> None:
    event = _public_event(
        {
            "event_type": "failed",
            "payload": {
                "transaction_id": "internal",
                "nested": {"request_id": "provider-id"},
                "reason": "api_key=ds-abcdefghijk and Authorization: Bearer sk-abcdefghijk",
            },
        }
    )
    assert "transaction_id" not in str(event)
    assert "request_id" not in str(event)
    assert "ds-abcdefghijk" not in str(event)
    assert "sk-abcdefghijk" not in str(event)
