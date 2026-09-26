"""Provider quota snapshots that never expose or persist credentials."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, cast
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, Field, ValidationError

from veripatch.config import Settings
from veripatch.credentials import load_api_key
from veripatch.domain import utc_now

_API_KEY_UNSET = object()


class BalanceInfo(BaseModel):
    currency: str = Field(min_length=3, max_length=3)
    total_balance: Decimal = Field(ge=0)
    granted_balance: Decimal = Field(ge=0)
    topped_up_balance: Decimal = Field(ge=0)


class QuotaSnapshot(BaseModel):
    provider: str
    supported: bool
    configured: bool
    is_available: bool | None = None
    low_balance: bool | None = None
    low_balance_threshold: Decimal | None = None
    balances: list[BalanceInfo] = Field(default_factory=list)
    checked_at: datetime = Field(default_factory=utc_now)
    error: str | None = None


class RefreshedTokenPair(BaseModel):
    access_token: str = Field(min_length=1)
    refresh_token: str = Field(min_length=1)
    expires_in: int = Field(gt=0)


def _unsupported_openai(api_key: str | None) -> QuotaSnapshot:
    return QuotaSnapshot(
        provider="openai",
        supported=False,
        configured=api_key is not None,
        error=(
            "官方 API Key 不提供剩余额度查询；请在 OpenAI 平台查看用量和费用。"
        ),
    )


def _proxy_balance_value(payload: Any) -> tuple[Decimal, str, bool] | None:
    if not isinstance(payload, dict):
        return None
    candidates = [payload]
    if isinstance(payload.get("data"), dict):
        candidates.append(payload["data"])
    for candidate in candidates:
        raw = next(
            (
                candidate[key]
                for key in ("balance", "remaining", "total_balance")
                if key in candidate
            ),
            None,
        )
        if raw is None:
            continue
        try:
            balance = Decimal(str(raw))
        except (InvalidOperation, TypeError, ValueError):
            continue
        currency = str(candidate.get("currency") or payload.get("currency") or "USD").upper()
        if len(currency) != 3:
            currency = "USD"
        active = bool(candidate.get("is_active", payload.get("is_active", True)))
        return balance, currency, active
    return None


def _newapi_token_balance(payload: Any) -> tuple[Decimal, str, bool] | None:
    """Map NewAPI's API-key usage response to a remaining USD balance."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    if data.get("object") != "token_usage":
        return None
    if bool(data.get("unlimited_quota")):
        return Decimal(0), "USD", True
    try:
        # NewAPI accounts in quota points; its default conversion is 500,000 points/USD.
        remaining = Decimal(str(data["total_available"])) / Decimal(500_000)
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None
    return max(remaining, Decimal(0)), "USD", remaining > 0


def _newapi_account_balance(payload: Any) -> tuple[Decimal, str, bool] | None:
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        return None
    data = payload["data"]
    try:
        remaining = Decimal(str(data["quota"])) / Decimal(500_000)
    except (InvalidOperation, KeyError, TypeError, ValueError):
        return None
    active = data.get("status", 1) == 1 and remaining > 0
    return max(remaining, Decimal(0)), "USD", active


def _pixel_account_balance(payload: Any) -> tuple[Decimal, str, bool] | None:
    if not isinstance(payload, dict):
        return None
    candidates = [payload]
    data = payload.get("data")
    if isinstance(data, dict):
        candidates.append(data)
        if isinstance(data.get("user"), dict):
            candidates.append(data["user"])
    if isinstance(payload.get("user"), dict):
        candidates.append(payload["user"])
    for candidate in candidates:
        if "balance" not in candidate:
            continue
        try:
            balance = Decimal(str(candidate["balance"]))
        except (InvalidOperation, TypeError, ValueError):
            continue
        active = candidate.get("status", 1) not in {0, "disabled"} and balance > 0
        return max(balance, Decimal(0)), "CNY", active
    return None


def _newapi_usage_url(base_url: str) -> str:
    parsed = urlsplit(base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunsplit((parsed.scheme, parsed.netloc, path + "/api/usage/token", "", ""))


def refresh_pixel_tokens(
    base_url: str, refresh_token: str, client: Any | None = None
) -> RefreshedTokenPair:
    owned_client = client is None
    resolved_client = client or httpx.Client(timeout=10, follow_redirects=False)
    url = _newapi_usage_url(base_url).removesuffix("/api/usage/token") + "/api/v1/auth/refresh"
    try:
        response = resolved_client.post(
            url,
            json={"refresh_token": refresh_token},
            headers={"Accept": "application/json", "User-Agent": "RAgent/0.1.0"},
        )
        if response.status_code != 200:
            raise ValueError(f"Pixel token refresh returned HTTP {response.status_code}")
        payload = response.json()
        if isinstance(payload, dict) and payload.get("code") == 0:
            payload = payload.get("data")
        return RefreshedTokenPair.model_validate(payload)
    finally:
        if owned_client:
            resolved_client.close()


def _openai_proxy_quota(
    settings: Settings,
    api_key: str | None,
    client: Any | None,
    quota_access_token: str | None = None,
    quota_user_id: str | None = None,
) -> QuotaSnapshot:
    if api_key is None:
        return QuotaSnapshot(
            provider="openai",
            supported=True,
            configured=False,
            error="OpenAI 中转站 API Key 尚未配置。",
        )
    owned_client = client is None
    resolved_client = client or httpx.Client(timeout=10)
    base_url = settings.openai_base_url.rstrip("/")
    urls = [
        _newapi_usage_url(base_url),
        base_url + "/user/balance",
    ]
    account_url = _newapi_usage_url(base_url).removesuffix("/usage/token") + "/user/self"
    pixel_url = _newapi_usage_url(base_url).removesuffix("/api/usage/token") + "/api/v1/auth/me"
    try:
        parsed: tuple[Decimal, str, bool] | None = None
        statuses: list[int] = []
        for index, url in enumerate(urls):
            try:
                response = resolved_client.get(
                    url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "RAgent/0.1.0",
                    },
                )
            except httpx.RequestError as exc:
                return QuotaSnapshot(
                    provider="openai",
                    supported=True,
                    configured=True,
                    error=f"中转站余额请求失败：{type(exc).__name__}。",
                )
            statuses.append(response.status_code)
            if response.status_code != 200:
                continue
            try:
                payload = response.json()
                parsed = (
                    _newapi_token_balance(payload) if index == 0 else _proxy_balance_value(payload)
                )
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                break
        if parsed is None and quota_access_token and quota_user_id:
            try:
                response = resolved_client.get(
                    account_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {quota_access_token}",
                        "New-Api-User": quota_user_id,
                        "User-Agent": "RAgent/0.1.0",
                    },
                )
                statuses.append(response.status_code)
                if response.status_code == 200:
                    parsed = _newapi_account_balance(response.json())
            except (httpx.RequestError, TypeError, ValueError) as exc:
                return QuotaSnapshot(
                    provider="openai",
                    supported=True,
                    configured=True,
                    error=f"NewAPI 账户额度请求失败：{type(exc).__name__}。",
                )
        if parsed is None and quota_access_token:
            try:
                response = resolved_client.get(
                    pixel_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {quota_access_token}",
                        "User-Agent": "RAgent/0.1.0",
                    },
                )
                statuses.append(response.status_code)
                if response.status_code == 200:
                    parsed = _pixel_account_balance(response.json())
            except (httpx.RequestError, TypeError, ValueError) as exc:
                return QuotaSnapshot(
                    provider="openai",
                    supported=True,
                    configured=True,
                    error=f"Pixel API 账户额度请求失败：{type(exc).__name__}。",
                )
        if parsed is None:
            missing_endpoints = statuses and all(status in {404, 405} for status in statuses)
            if missing_endpoints and not quota_access_token:
                error = "模型密钥未开放余额查询；可配置中转站后台访问令牌。"
            elif missing_endpoints:
                error = "中转站未提供 NewAPI 或标准余额接口。"
            elif 401 in statuses or 403 in statuses:
                error = "余额接口鉴权失败；模型密钥可能不具备额度查询权限。"
            elif any(status != 200 for status in statuses):
                error = "中转站余额接口返回 HTTP " + "/".join(map(str, statuses)) + "。"
            else:
                error = "中转站余额响应格式不受支持。"
            return QuotaSnapshot(provider="openai", supported=True, configured=True, error=error)
        balance, currency, active = parsed
        return QuotaSnapshot(
            provider="openai",
            supported=True,
            configured=True,
            is_available=active,
            low_balance=not active or balance <= 0,
            balances=[
                BalanceInfo(
                    currency=currency,
                    total_balance=balance,
                    granted_balance=Decimal(0),
                    topped_up_balance=balance,
                )
            ],
        )
    finally:
        if owned_client:
            resolved_client.close()


def get_provider_quota(
    provider: str,
    settings: Settings,
    *,
    client: Any | None = None,
    api_key: str | None | object = _API_KEY_UNSET,
    quota_access_token: str | None = None,
    quota_user_id: str | None = None,
) -> QuotaSnapshot:
    if provider not in {"openai", "openai_official", "deepseek"}:
        raise ValueError(f"Unknown quota provider: {provider}")
    resolved_api_key = (
        load_api_key(provider) if api_key is _API_KEY_UNSET else cast(str | None, api_key)
    )
    if provider == "openai_official":
        return _unsupported_openai(resolved_api_key).model_copy(
            update={"provider": "openai_official"}
        )
    if provider == "openai":
        if settings.openai_base_url.rstrip("/") == "https://api.openai.com/v1":
            return _unsupported_openai(resolved_api_key)
        return _openai_proxy_quota(
            settings,
            resolved_api_key,
            client,
            quota_access_token=quota_access_token,
            quota_user_id=quota_user_id,
        )

    threshold = Decimal(str(settings.deepseek_low_balance))
    if resolved_api_key is None:
        return QuotaSnapshot(
            provider=provider,
            supported=True,
            configured=False,
            low_balance_threshold=threshold,
            error="DeepSeek API key is not configured.",
        )

    owned_client = client is None
    resolved_client = client or httpx.Client(timeout=10)
    try:
        try:
            response = resolved_client.get(
                settings.deepseek_base_url.rstrip("/") + "/user/balance",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {resolved_api_key}",
                },
            )
        except httpx.RequestError as exc:
            return QuotaSnapshot(
                provider=provider,
                supported=True,
                configured=True,
                low_balance_threshold=threshold,
                error=f"DeepSeek balance request failed: {type(exc).__name__}.",
            )
        if response.status_code != 200:
            error_by_status = {
                401: "DeepSeek API Key 认证失败，请在 API 配置中更新密钥。",
                402: "DeepSeek API 余额不足，请充值后重试。",
                429: "DeepSeek 请求过于频繁，请稍后重试。",
            }
            return QuotaSnapshot(
                provider=provider,
                supported=True,
                configured=True,
                low_balance_threshold=threshold,
                error=error_by_status.get(
                    response.status_code,
                    f"DeepSeek 额度请求返回 HTTP {response.status_code}。",
                ),
            )
        try:
            payload = response.json()
            balances = [BalanceInfo.model_validate(item) for item in payload["balance_infos"]]
            is_available = bool(payload["is_available"])
        except (KeyError, TypeError, ValueError, ValidationError):
            return QuotaSnapshot(
                provider=provider,
                supported=True,
                configured=True,
                low_balance_threshold=threshold,
                error="DeepSeek returned an invalid balance response.",
            )
        low_balance = not is_available or (
            bool(balances) and all(item.total_balance <= threshold for item in balances)
        )
        return QuotaSnapshot(
            provider=provider,
            supported=True,
            configured=True,
            is_available=is_available,
            low_balance=low_balance,
            low_balance_threshold=threshold,
            balances=balances,
        )
    finally:
        if owned_client:
            resolved_client.close()
