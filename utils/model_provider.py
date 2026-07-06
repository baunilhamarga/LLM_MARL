"""Provider-neutral chat completions and usage normalization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
import os
from typing import Any, Mapping, Optional


PROVIDER_CONFIGS = {
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "requires_api_key": True,
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "base_url": "https://api.groq.com/openai/v1",
        "requires_api_key": True,
    },
    "openai-compatible": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "requires_api_key": False,
    },
}


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert SDK objects to dictionaries without assuming a specific SDK."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(exclude_none=True)
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return value.dict(exclude_none=True)
        except Exception:
            pass
    try:
        return vars(value)
    except (TypeError, AttributeError):
        return {}


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, Number]:
    """Flatten every numeric leaf while preserving provider-specific fields."""
    output: dict[str, Number] = {}
    mapping = _as_dict(value)
    if mapping:
        items = mapping.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    elif isinstance(value, Number) and not isinstance(value, bool) and prefix:
        return {prefix: value}
    else:
        return output

    for key, item in items:
        name = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten_numeric(item, name))
    return output


def response_metadata(response: Any) -> dict[str, Any]:
    """Collect stable, JSON-safe metadata useful for reproducibility."""
    metadata = {}
    for name in ("id", "created", "model", "system_fingerprint", "service_tier"):
        value = getattr(response, name, None)
        if value is not None and isinstance(value, (str, int, float, bool)):
            metadata[name] = value
    return metadata


@dataclass
class CompletionResult:
    content: str
    usage: dict[str, Number]
    metadata: dict[str, Any]
    usage_available: bool


class ChatModelClient:
    """Small adapter around OpenAI-compatible Chat Completions endpoints."""

    def __init__(
        self,
        provider: str = "openai",
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        client: Any = None,
    ):
        if provider not in PROVIDER_CONFIGS:
            supported = ", ".join(PROVIDER_CONFIGS)
            raise ValueError(f"Unsupported provider '{provider}'. Choose from: {supported}")

        config = PROVIDER_CONFIGS[provider]
        self.provider = provider
        self.base_url = base_url or config["base_url"]
        self.api_key_env = api_key_env or config["api_key_env"]
        self.requires_api_key = config["requires_api_key"]
        self._client = client

        if provider == "openai-compatible" and not self.base_url:
            raise ValueError("--base_url is required for provider 'openai-compatible'")

    def _create_client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install openai>=1.0.0 to use model providers") from exc

        api_key = os.environ.get(self.api_key_env)
        if self.requires_api_key and not api_key:
            raise RuntimeError(
                f"Provider '{self.provider}' requires the {self.api_key_env} environment variable"
            )

        kwargs = {"api_key": api_key or "not-required"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    @property
    def client(self):
        if self._client is None:
            self._client = self._create_client()
        return self._client

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
    ) -> CompletionResult:
        request = {"model": model, "messages": messages}
        if temperature is not None:
            request["temperature"] = temperature

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        usage_obj = getattr(response, "usage", None)
        usage = flatten_numeric(usage_obj)
        for field in ("usage_breakdown", "x_groq"):
            usage.update(flatten_numeric(getattr(response, field, None), field))
        return CompletionResult(
            content=content,
            usage=usage,
            metadata=response_metadata(response),
            usage_available=usage_obj is not None,
        )

    @staticmethod
    def rejects_temperature(error: Exception) -> bool:
        """Detect provider errors that can be retried without temperature."""
        message = str(error).lower()
        status_code = getattr(error, "status_code", None)
        mentions_temperature = "temperature" in message
        unsupported = any(term in message for term in ("unsupported", "not support", "not available"))
        return (status_code in (None, 400)) and mentions_temperature and unsupported
