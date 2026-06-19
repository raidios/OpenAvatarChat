from dataclasses import asdict, is_dataclass
from typing import Any, Mapping


_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def _is_sensitive_key(key: Any) -> bool:
    text = str(key).lower()
    return any(part in text for part in _SENSITIVE_KEY_PARTS)


def redact_sensitive_config_for_log(value: Any) -> Any:
    """Return a log-safe copy of config-like data with secrets redacted."""
    if isinstance(value, Mapping):
        return {
            key: "<redacted>" if _is_sensitive_key(key) else redact_sensitive_config_for_log(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        redacted = [redact_sensitive_config_for_log(item) for item in value]
        return tuple(redacted) if isinstance(value, tuple) else redacted

    if hasattr(value, "model_dump"):
        return redact_sensitive_config_for_log(value.model_dump())

    if is_dataclass(value) and not isinstance(value, type):
        return redact_sensitive_config_for_log(asdict(value))

    return value
