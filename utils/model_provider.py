"""Provider-neutral chat completions and usage normalization."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number
import os
import time
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
    "local": {
        "api_key_env": None,
        "base_url": None,
        "requires_api_key": False,
    },
    "openai-compatible": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
        "requires_api_key": False,
    },
}

MAX_AGGREGATED_METRICS = {
    "cuda_memory_allocated_mb",
    "cuda_memory_reserved_mb",
    "cuda_peak_memory_allocated_mb",
}


def aggregate_metrics(target: dict[str, Number], metrics: Mapping[str, Number]) -> None:
    """Aggregate counters/timers by sum and resource gauges by maximum."""
    for key, value in metrics.items():
        if key in MAX_AGGREGATED_METRICS:
            target[key] = max(target.get(key, value), value)
        else:
            target[key] = target.get(key, 0) + value


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
        model_path: Optional[str] = None,
        model_cache_dir: Optional[str] = None,
        local_dtype: str = "float16",
        max_completion_tokens: Optional[int] = None,
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
        self.model_path = model_path
        self.model_cache_dir = model_cache_dir
        self.local_dtype = local_dtype
        self.max_completion_tokens = max_completion_tokens
        self.setup_metrics: dict[str, Any] = {}
        self._client = client

        if provider == "openai-compatible" and not self.base_url:
            raise ValueError("--base_url is required for provider 'openai-compatible'")

    def _create_client(self):
        if self.provider == "local":
            raise RuntimeError("The local model must be prepared with prepare(model) first")

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

    def _prepare_local(self, model: str):
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local inference requires torch, transformers, accelerate, and safetensors; "
                "install requirements-local.txt first"
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError(
                "Local inference requires a CUDA GPU. Run it inside the allocated GPU compute node."
            )

        dtype_by_name = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.local_dtype not in dtype_by_name:
            raise ValueError(f"Unsupported local dtype: {self.local_dtype}")

        source = self.model_path or model
        load_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            cache_dir=self.model_cache_dir,
            local_files_only=True,
        )
        local_model = AutoModelForCausalLM.from_pretrained(
            source,
            cache_dir=self.model_cache_dir,
            local_files_only=True,
            device_map="auto",
            torch_dtype=dtype_by_name[self.local_dtype],
            low_cpu_mem_usage=True,
        )
        local_model.eval()
        torch.cuda.synchronize()

        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)
        parameter_count = sum(parameter.numel() for parameter in local_model.parameters())
        self.setup_metrics = {
            "load_time_seconds": time.perf_counter() - load_started,
            "model_source": source,
            "model_parameter_count": parameter_count,
            "dtype": str(next(local_model.parameters()).dtype),
            "device": str(next(local_model.parameters()).device),
            "cuda_device_name": properties.name,
            "cuda_device_total_memory_mb": properties.total_memory / (1024 ** 2),
            "cuda_memory_allocated_after_load_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "cuda_memory_reserved_after_load_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "cuda_peak_memory_allocated_during_load_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        }
        self._client = {
            "torch": torch,
            "tokenizer": tokenizer,
            "model": local_model,
        }

    def prepare(self, model: str):
        """Validate a remote client or load the local model before the experiment."""
        if self.provider == "local":
            if self._client is None:
                self._prepare_local(model)
        else:
            self.client
        return self

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
        if self.provider == "local":
            return self._complete_local(model=model, messages=messages, temperature=temperature)

        request = {"model": model, "messages": messages}
        if self.max_completion_tokens is not None:
            request["max_completion_tokens"] = self.max_completion_tokens
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

    def _complete_local(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: Optional[float],
    ) -> CompletionResult:
        if self._client is None:
            self._prepare_local(model)

        torch = self._client["torch"]
        tokenizer = self._client["tokenizer"]
        local_model = self._client["model"]
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        input_ids = input_ids.to(next(local_model.parameters()).device)
        attention_mask = torch.ones_like(input_ids)

        generation_kwargs = {
            "max_new_tokens": self.max_completion_tokens or 256,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if temperature is not None and temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": temperature})
        else:
            generation_kwargs["do_sample"] = False

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output_ids = local_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )
        torch.cuda.synchronize()
        generation_time = time.perf_counter() - generation_started

        generated_ids = output_ids[0, input_ids.shape[-1]:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        prompt_tokens = int(input_ids.shape[-1])
        completion_tokens = int(generated_ids.shape[-1])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_time_seconds": generation_time,
            "cuda_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
            "cuda_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
            "cuda_peak_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 ** 2),
        }
        metadata = {
            "model": model,
            "model_source": self.model_path or model,
            "finish_reason": "length" if completion_tokens >= generation_kwargs["max_new_tokens"] else "stop",
        }
        return CompletionResult(
            content=content,
            usage=usage,
            metadata=metadata,
            usage_available=True,
        )

    @staticmethod
    def rejects_temperature(error: Exception) -> bool:
        """Detect provider errors that can be retried without temperature."""
        message = str(error).lower()
        status_code = getattr(error, "status_code", None)
        mentions_temperature = "temperature" in message
        unsupported = any(term in message for term in ("unsupported", "not support", "not available"))
        return (status_code in (None, 400)) and mentions_temperature and unsupported
