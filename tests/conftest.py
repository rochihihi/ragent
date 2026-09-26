from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_credential_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never allow automated tests to read or mutate a user's credential vault."""
    monkeypatch.setenv("VERIPATCH_CREDENTIAL_VAULT", str(tmp_path / "credentials.dpapi"))
