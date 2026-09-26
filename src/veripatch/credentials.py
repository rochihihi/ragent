"""Provider API keys resolved without entering run state or event storage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SERVICE_NAME = "veripatch"
PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "openai_official": "OPENAI_OFFICIAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "openai_quota": "VERIPATCH_OPENAI_QUOTA_TOKEN",
    "openai_quota_refresh": "VERIPATCH_OPENAI_QUOTA_REFRESH_TOKEN",
}


def _vault_path() -> Path:
    return Path(os.getenv("VERIPATCH_CREDENTIAL_VAULT", "runs/credentials.dpapi"))


def _protect_windows_data(data: bytes, *, decrypt: bool = False) -> bytes:
    if os.name != "nt":
        raise RuntimeError("Windows DPAPI is unavailable")
    import ctypes
    from ctypes import wintypes

    class DataBlob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]

    input_buffer = (ctypes.c_byte * len(data)).from_buffer_copy(data)
    input_blob = DataBlob(len(data), ctypes.cast(input_buffer, ctypes.POINTER(ctypes.c_byte)))
    output_blob = DataBlob()
    win_dll: Any = ctypes.__dict__["WinDLL"]
    crypt32 = win_dll("Crypt32.dll", use_last_error=True)
    kernel32 = win_dll("Kernel32.dll", use_last_error=True)
    crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(DataBlob),
        wintypes.LPCWSTR,
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(DataBlob),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(DataBlob),
    ]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    if decrypt:
        succeeded = crypt32.CryptUnprotectData(
            ctypes.byref(input_blob),
            None,
            None,
            None,
            None,
            0x1,
            ctypes.byref(output_blob),
        )
    else:
        succeeded = crypt32.CryptProtectData(
            ctypes.byref(input_blob),
            "VeriPatch credentials",
            None,
            None,
            None,
            0x1,
            ctypes.byref(output_blob),
        )
    if not succeeded:
        get_last_error: Any = ctypes.__dict__["get_last_error"]
        raise OSError(get_last_error(), "Windows DPAPI operation failed")
    try:
        return ctypes.string_at(output_blob.data, output_blob.size)
    finally:
        kernel32.LocalFree(ctypes.cast(output_blob.data, ctypes.c_void_p))


def _load_vault() -> dict[str, str]:
    path = _vault_path()
    if not path.is_file():
        return {}
    decrypted = _protect_windows_data(path.read_bytes(), decrypt=True)
    payload: Any = json.loads(decrypted.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Credential vault has an invalid shape")
    return {
        provider: value
        for provider, value in payload.items()
        if provider in PROVIDER_ENV and isinstance(value, str) and value
    }


def _save_vault(values: dict[str, str]) -> None:
    path = _vault_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _protect_windows_data(json.dumps(values, separators=(",", ":")).encode("utf-8"))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encrypted)
    temporary.replace(path)


def _stored_credential(provider: str) -> tuple[str | None, str]:
    try:
        import keyring

        value = keyring.get_password(SERVICE_NAME, provider)
        if value:
            return value, "keyring"
    except Exception:
        pass
    try:
        value = _load_vault().get(provider)
        return value, "encrypted_vault" if value else "none"
    except Exception:
        return None, "none"


def _provider(provider: str) -> str:
    if provider not in PROVIDER_ENV:
        raise ValueError(f"Unknown credential provider: {provider}")
    return provider


def load_api_key(provider: str) -> str | None:
    provider = _provider(provider)
    environment_value = os.getenv(PROVIDER_ENV[provider])
    if environment_value:
        return environment_value
    return _stored_credential(provider)[0]


def save_api_key(provider: str, api_key: str) -> None:
    provider = _provider(provider)
    if not api_key.strip():
        raise ValueError("API key cannot be empty")
    keyring_failed = False
    try:
        import keyring

        keyring.set_password(SERVICE_NAME, provider, api_key.strip())
        if keyring.get_password(SERVICE_NAME, provider) == api_key.strip():
            return
        keyring_failed = True
    except Exception:
        keyring_failed = True
    if keyring_failed:
        try:
            values = _load_vault()
            values[provider] = api_key.strip()
            _save_vault(values)
            if _load_vault().get(provider) == api_key.strip():
                return
        except Exception as exc:
            raise RuntimeError(
                f"Credential storage is unavailable; set {PROVIDER_ENV[provider]} instead"
            ) from exc
    raise RuntimeError(f"Credential storage is unavailable; set {PROVIDER_ENV[provider]} instead")


def delete_api_key(provider: str) -> bool:
    provider = _provider(provider)
    removed = False
    try:
        import keyring

        existing = keyring.get_password(SERVICE_NAME, provider)
        if existing is not None:
            keyring.delete_password(SERVICE_NAME, provider)
            removed = True
    except Exception:
        pass
    try:
        values = _load_vault()
        if provider in values:
            del values[provider]
            _save_vault(values)
            removed = True
    except Exception:
        pass
    return removed


def credential_status() -> dict[str, dict[str, str | bool]]:
    result: dict[str, dict[str, str | bool]] = {}
    for provider, environment_name in PROVIDER_ENV.items():
        if os.getenv(environment_name):
            source = "environment"
            configured = True
        else:
            value, source = _stored_credential(provider)
            configured = value is not None
        result[provider] = {"configured": configured, "source": source}
    return result
