import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from veripatch import credentials


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, provider: str) -> str | None:
        return self.values.get((service, provider))

    def set_password(self, service: str, provider: str, value: str) -> None:
        self.values[(service, provider)] = value

    def delete_password(self, service: str, provider: str) -> None:
        del self.values[(service, provider)]


def test_environment_precedes_keyring_and_status_never_returns_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeKeyring()
    fake.set_password("veripatch", "openai", "keyring-secret")
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    assert credentials.load_api_key("openai") == "environment-secret"
    status = credentials.credential_status()
    assert status["openai"] == {"configured": True, "source": "environment"}
    assert "secret" not in repr(status)


def test_save_load_and_delete_keyring_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeKeyring()
    monkeypatch.setitem(sys.modules, "keyring", fake)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    credentials.save_api_key("deepseek", "  ds-key  ")
    assert credentials.load_api_key("deepseek") == "ds-key"
    assert credentials.credential_status()["deepseek"]["source"] == "keyring"
    assert credentials.delete_api_key("deepseek")
    assert not credentials.delete_api_key("deepseek")


def test_windows_encrypted_vault_falls_back_when_keyring_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vault: dict[str, str] = {}
    broken = SimpleNamespace(
        get_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
        set_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
        delete_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
    )

    def save_vault(values: dict[str, str]) -> None:
        vault.clear()
        vault.update(values)

    monkeypatch.setitem(sys.modules, "keyring", broken)
    monkeypatch.setattr(credentials, "_load_vault", lambda: dict(vault))
    monkeypatch.setattr(credentials, "_save_vault", save_vault)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    credentials.save_api_key("deepseek", "fallback-key")
    assert credentials.load_api_key("deepseek") == "fallback-key"
    assert credentials.credential_status()["deepseek"] == {
        "configured": True,
        "source": "encrypted_vault",
    }
    assert credentials.delete_api_key("deepseek") is True
    assert credentials.load_api_key("deepseek") is None


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI only")
def test_windows_encrypted_vault_round_trip_is_not_plaintext(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    vault_path = tmp_path / "credentials.dpapi"
    broken = SimpleNamespace(
        get_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
        set_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
        delete_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
    )
    monkeypatch.setitem(sys.modules, "keyring", broken)
    monkeypatch.setenv("VERIPATCH_CREDENTIAL_VAULT", str(vault_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    credentials.save_api_key("openai", "dpapi-secret-marker")
    assert credentials.load_api_key("openai") == "dpapi-secret-marker"
    assert b"dpapi-secret-marker" not in vault_path.read_bytes()
    assert credentials.delete_api_key("openai") is True
    assert credentials.load_api_key("openai") is None


def test_credentials_reject_unknown_empty_and_unavailable_keyring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        credentials.load_api_key("other")
    with pytest.raises(ValueError, match="empty"):
        credentials.save_api_key("openai", "  ")
    broken = SimpleNamespace(
        get_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
        set_password=lambda *_: (_ for _ in ()).throw(RuntimeError("locked")),
    )
    monkeypatch.setitem(sys.modules, "keyring", broken)
    monkeypatch.setattr(credentials, "_load_vault", lambda: {})
    monkeypatch.setattr(
        credentials,
        "_save_vault",
        lambda _values: (_ for _ in ()).throw(RuntimeError("vault locked")),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert credentials.load_api_key("openai") is None
    assert credentials.delete_api_key("openai") is False
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        credentials.save_api_key("openai", "key")
