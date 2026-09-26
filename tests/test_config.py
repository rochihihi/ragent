from pathlib import Path

import pytest

from veripatch.config import Settings


def test_settings_from_env_maps_provider_and_budget_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VERIPATCH_DEEPSEEK_MODEL", "deepseek-v4-pro")
    monkeypatch.setenv("VERIPATCH_DEEPSEEK_BASE_URL", "https://example.invalid")
    monkeypatch.setenv("VERIPATCH_OPENAI_BASE_URL", "https://proxy.example/v1")
    monkeypatch.setenv("VERIPATCH_MAX_MODEL_CALLS", "3")
    monkeypatch.setenv("VERIPATCH_MAX_INPUT_TOKENS", "100")
    monkeypatch.setenv("VERIPATCH_MAX_OUTPUT_TOKENS", "20")
    monkeypatch.setenv("VERIPATCH_DEEPSEEK_LOW_BALANCE", "2.5")
    monkeypatch.setenv("VERIPATCH_DATABASE_PATH", str(tmp_path / "runs.db"))
    monkeypatch.setenv("VERIPATCH_TEST_RUNNER", "local")
    settings = Settings.from_env()
    assert settings.deepseek_model == "deepseek-v4-pro"
    assert settings.deepseek_base_url == "https://example.invalid"
    assert settings.openai_base_url == "https://proxy.example/v1"
    assert settings.max_model_calls == 3
    assert settings.deepseek_low_balance == 2.5
    assert settings.database_path == tmp_path / "runs.db"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("VERIPATCH_MAX_STEPS", "zero", "integer"),
        ("VERIPATCH_MAX_STEPS", "0", "positive"),
        ("VERIPATCH_REASONING_EFFORT", "extreme", "must be one of"),
        ("VERIPATCH_TEST_RUNNER", "host", "local.*docker"),
        ("VERIPATCH_DEEPSEEK_LOW_BALANCE", "many", "number"),
        ("VERIPATCH_DEEPSEEK_LOW_BALANCE", "-1", "negative"),
    ],
)
def test_settings_reject_invalid_environment(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, message: str
) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=message):
        Settings.from_env()
