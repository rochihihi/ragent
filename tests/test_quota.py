from decimal import Decimal
from types import SimpleNamespace

import httpx
import pytest

from veripatch import quota
from veripatch.config import Settings


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def json(self) -> object:
        return self.payload


class FakeClient:
    def __init__(
        self,
        payload: object | None = None,
        *,
        status_code: int = 200,
        error: Exception | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.error = error
        self.url: str | None = None
        self.headers: dict[str, str] | None = None

    def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
        self.url = url
        self.headers = headers
        if self.error:
            raise self.error
        return FakeResponse(self.payload, self.status_code)


def _payload(total: str = "12.50", *, available: bool = True) -> dict:
    return {
        "is_available": available,
        "balance_infos": [
            {
                "currency": "CNY",
                "total_balance": total,
                "granted_balance": "2.50",
                "topped_up_balance": "10.00",
            }
        ],
    }


def test_deepseek_quota_maps_balance_without_exposing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quota, "load_api_key", lambda provider: "ds-private-value")
    client = FakeClient(_payload())
    snapshot = quota.get_provider_quota("deepseek", Settings(), client=client)
    assert snapshot.is_available is True
    assert snapshot.low_balance is False
    assert snapshot.balances[0].total_balance == Decimal("12.50")
    assert client.url == "https://api.deepseek.com/user/balance"
    assert client.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer ds-private-value",
    }
    assert "ds-private-value" not in snapshot.model_dump_json()


def test_deepseek_quota_marks_low_or_unavailable_balance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quota, "load_api_key", lambda provider: "key")
    low = quota.get_provider_quota(
        "deepseek", Settings(deepseek_low_balance=5), client=FakeClient(_payload("5.00"))
    )
    assert low.low_balance is True
    unavailable = quota.get_provider_quota(
        "deepseek", Settings(), client=FakeClient(_payload("10", available=False))
    )
    assert unavailable.low_balance is True


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"balance": "18.75", "currency": "CNY", "is_active": True}, "18.75"),
        ({"data": {"remaining": 7.5, "currency": "USD"}}, "7.5"),
    ],
)
def test_openai_proxy_quota_uses_generic_balance_endpoint(payload: dict, expected: str) -> None:
    client = FakeClient(payload)
    snapshot = quota.get_provider_quota(
        "openai",
        Settings(openai_base_url="https://proxy.example/v1"),
        client=client,
        api_key="proxy-private-key",
    )
    assert snapshot.supported is True
    assert snapshot.is_available is True
    assert snapshot.balances[0].total_balance == Decimal(expected)
    assert client.url == "https://proxy.example/v1/user/balance"
    assert client.headers == {
        "Accept": "application/json",
        "Authorization": "Bearer proxy-private-key",
        "User-Agent": "RAgent/0.1.0",
    }
    assert "proxy-private-key" not in snapshot.model_dump_json()


def test_openai_proxy_quota_reports_unsupported_response() -> None:
    snapshot = quota.get_provider_quota(
        "openai",
        Settings(openai_base_url="https://proxy.example/v1"),
        client=FakeClient({"unexpected": True}),
        api_key="key",
    )
    assert snapshot.supported is True
    assert snapshot.error == "中转站余额响应格式不受支持。"


def test_openai_proxy_quota_supports_newapi_token_usage() -> None:
    client = FakeClient(
        {
            "code": True,
            "data": {
                "object": "token_usage",
                "total_granted": 1_000_000,
                "total_used": 125_000,
                "total_available": 875_000,
                "unlimited_quota": False,
            },
        }
    )
    snapshot = quota.get_provider_quota(
        "openai",
        Settings(openai_base_url="https://newapi.example/v1"),
        client=client,
        api_key="sk-newapi-token",
    )
    assert snapshot.is_available is True
    assert snapshot.balances[0].total_balance == Decimal("1.75")
    assert client.url == "https://newapi.example/api/usage/token"


def test_openai_proxy_quota_supports_newapi_account_credentials() -> None:
    class AccountClient(FakeClient):
        def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            self.url = url
            self.headers = headers
            if url.endswith("/api/user/self"):
                return FakeResponse({"success": True, "data": {"quota": 750_000, "status": 1}})
            return FakeResponse({}, 404)

    client = AccountClient()
    snapshot = quota.get_provider_quota(
        "openai",
        Settings(openai_base_url="https://newapi.example/v1"),
        client=client,
        api_key="sk-model-key",
        quota_access_token="account-access-token",
        quota_user_id="7",
    )
    assert snapshot.balances[0].total_balance == Decimal("1.5")
    assert client.url == "https://newapi.example/api/user/self"
    assert client.headers is not None
    assert client.headers["New-Api-User"] == "7"
    assert "account-access-token" not in snapshot.model_dump_json()


def test_openai_proxy_quota_supports_pixel_account_token() -> None:
    class PixelClient(FakeClient):
        def get(self, url: str, *, headers: dict[str, str]) -> FakeResponse:
            self.url = url
            self.headers = headers
            if url.endswith("/api/v1/auth/me"):
                return FakeResponse({"data": {"user": {"balance": "28.75", "status": 1}}})
            return FakeResponse({}, 404)

    client = PixelClient()
    snapshot = quota.get_provider_quota(
        "openai",
        Settings(openai_base_url="https://pixel.example/v1"),
        client=client,
        api_key="sk-model-key",
        quota_access_token="pixel-login-token",
    )
    assert snapshot.is_available is True
    assert snapshot.balances[0].currency == "CNY"
    assert snapshot.balances[0].total_balance == Decimal("28.75")
    assert client.url == "https://pixel.example/api/v1/auth/me"
    assert "pixel-login-token" not in snapshot.model_dump_json()


def test_pixel_refresh_rotates_token_pair() -> None:
    class RefreshClient:
        def __init__(self) -> None:
            self.url = ""
            self.body: dict[str, str] = {}

        def post(
            self,
            url: str,
            *,
            json: dict[str, str],
            headers: dict[str, str],
        ) -> FakeResponse:
            self.url = url
            self.body = json
            assert "Authorization" not in headers
            return FakeResponse(
                {
                    "code": 0,
                    "data": {
                        "access_token": "next-access",
                        "refresh_token": "next-refresh",
                        "expires_in": 3600,
                    },
                }
            )

    client = RefreshClient()
    pair = quota.refresh_pixel_tokens("https://pixel.example/v1", "old-refresh", client=client)
    assert pair.access_token == "next-access"
    assert pair.refresh_token == "next-refresh"
    assert client.body == {"refresh_token": "old-refresh"}
    assert client.url == "https://pixel.example/api/v1/auth/refresh"


def test_quota_reports_missing_key_unsupported_provider_and_openai_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(quota, "load_api_key", lambda provider: None)
    missing = quota.get_provider_quota("deepseek", Settings())
    assert missing.configured is False
    assert "not configured" in (missing.error or "")
    openai = quota.get_provider_quota("openai", Settings())
    assert openai.supported is False
    assert "官方 API Key 不提供剩余额度查询" in (openai.error or "")
    with pytest.raises(ValueError, match="Unknown quota provider"):
        quota.get_provider_quota("other", Settings())


@pytest.mark.parametrize(
    "client",
    [
        FakeClient(status_code=401),
        FakeClient({"unexpected": True}),
        FakeClient(error=httpx.ConnectError("offline", request=httpx.Request("GET", "https://x"))),
    ],
)
def test_quota_converts_provider_failures_to_safe_snapshots(
    monkeypatch: pytest.MonkeyPatch, client: FakeClient
) -> None:
    monkeypatch.setattr(quota, "load_api_key", lambda provider: "ds-do-not-leak")
    snapshot = quota.get_provider_quota("deepseek", Settings(), client=client)
    assert snapshot.error is not None
    assert "ds-do-not-leak" not in snapshot.model_dump_json()


def test_quota_explains_authentication_and_balance_errors() -> None:
    settings = Settings()
    unauthorized = quota.get_provider_quota(
        "deepseek", settings, client=FakeClient(status_code=401), api_key="key"
    )
    insufficient = quota.get_provider_quota(
        "deepseek", settings, client=FakeClient(status_code=402), api_key="key"
    )
    assert unauthorized.error == "DeepSeek API Key 认证失败，请在 API 配置中更新密钥。"
    assert insufficient.error == "DeepSeek API 余额不足，请充值后重试。"


def test_owned_http_client_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient(_payload())
    fake.close = SimpleNamespace(called=False)

    def close() -> None:
        fake.close.called = True

    fake.close = close
    monkeypatch.setattr(quota, "load_api_key", lambda provider: "key")
    monkeypatch.setattr(quota.httpx, "Client", lambda timeout: fake)
    quota.get_provider_quota("deepseek", Settings())
    assert fake.close.called
