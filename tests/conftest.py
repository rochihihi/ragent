from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_credential_vault(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never allow automated tests to read or mutate a user's credential vault."""
    monkeypatch.setenv("VERIPATCH_CREDENTIAL_VAULT", str(tmp_path / "credentials.dpapi"))
    # Legacy functional tests explicitly opt out; sandbox tests enable required
    # in this isolated location. Never install accounts or touch user settings.
    policy = tmp_path_factory.mktemp("sandbox-host") / "sandbox.json"
    policy.write_text('{"mode":"off"}', encoding="utf-8")
    monkeypatch.setenv("RAGENT_SANDBOX_CONFIG", str(policy))
