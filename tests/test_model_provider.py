import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from utils.model_provider import ChatModelClient, aggregate_metrics, flatten_numeric

try:
    from dragonTextEnv import ChatAgent
except ImportError:
    ChatAgent = None


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.response


class TemperatureRejectingCompletions(FakeCompletions):
    def create(self, **kwargs):
        self.requests.append(kwargs)
        if "temperature" in kwargs:
            error = RuntimeError("temperature is unsupported for this model")
            error.status_code = 400
            raise error
        return self.response


def fake_client(response):
    completions = FakeCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


class UsageNormalizationTests(unittest.TestCase):
    def test_flattens_sdk_objects_nested_mappings_and_lists(self):
        usage = SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=4,
            prompt_tokens_details={"cached_tokens": 2},
            models=[{"usage": {"total_tokens": 14}}],
            ignored=None,
        )

        self.assertEqual(
            flatten_numeric(usage),
            {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_tokens_details.cached_tokens": 2,
                "models.0.usage.total_tokens": 14,
            },
        )

    def test_missing_usage_does_not_fail_completion(self):
        response = SimpleNamespace(
            id="response-1",
            model="local-model",
            choices=[SimpleNamespace(message=SimpleNamespace(content="move to room 0"))],
            usage=None,
        )
        client, _ = fake_client(response)
        adapter = ChatModelClient(provider="openai-compatible", base_url="http://localhost:8000/v1", client=client)

        result = adapter.complete(model="local-model", messages=[], temperature=0)

        self.assertEqual(result.content, "move to room 0")
        self.assertEqual(result.usage, {})
        self.assertFalse(result.usage_available)

    def test_groq_uses_its_key_and_openai_compatible_endpoint(self):
        sentinel = object()
        adapter = ChatModelClient(provider="groq")
        constructor = Mock(return_value=sentinel)
        fake_openai = SimpleNamespace(OpenAI=constructor)
        with patch.dict(os.environ, {"GROQ_API_KEY": "secret"}), patch.dict(
            sys.modules, {"openai": fake_openai}
        ):
            self.assertIs(adapter.client, sentinel)

        constructor.assert_called_once_with(
            api_key="secret",
            base_url="https://api.groq.com/openai/v1",
        )

    def test_temperature_rejection_detection_is_specific(self):
        error = RuntimeError("temperature is unsupported for this model")
        self.assertTrue(ChatModelClient.rejects_temperature(error))
        self.assertFalse(ChatModelClient.rejects_temperature(RuntimeError("model not found")))

    def test_resource_gauges_use_maximum_while_tokens_use_sum(self):
        total = {}
        aggregate_metrics(total, {"prompt_tokens": 10, "cuda_peak_memory_allocated_mb": 100})
        aggregate_metrics(total, {"prompt_tokens": 20, "cuda_peak_memory_allocated_mb": 90})

        self.assertEqual(total["prompt_tokens"], 30)
        self.assertEqual(total["cuda_peak_memory_allocated_mb"], 100)

    def test_local_provider_configuration_requires_no_api_key(self):
        adapter = ChatModelClient(
            provider="local",
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            model_cache_dir="/models",
        )

        self.assertFalse(adapter.requires_api_key)
        self.assertEqual(adapter.model_path, "meta-llama/Llama-3.1-8B-Instruct")


@unittest.skipIf(ChatAgent is None, "project runtime dependencies are not installed")
class ChatAgentMetricsTests(unittest.TestCase):
    def test_agent_accumulates_usage_and_request_metrics(self):
        response = SimpleNamespace(
            id="response-1",
            model="test-model",
            system_fingerprint="backend-v1",
            choices=[SimpleNamespace(message=SimpleNamespace(content="inspect bomb"))],
            usage={
                "prompt_tokens": 12,
                "completion_tokens": 3,
                "total_tokens": 15,
                "total_time": 0.05,
            },
        )
        client, _ = fake_client(response)
        adapter = ChatModelClient(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            client=client,
        )
        agent = ChatAgent(
            model="test-model",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model_client=adapter,
            allow_comm=False,
            log_chat=False,
        )

        self.assertEqual(agent.makeAPIcall(), "inspect bomb")
        self.assertEqual(agent.total_usage["prompt_tokens"], 12)
        self.assertEqual(agent.total_usage["completion_tokens"], 3)
        self.assertEqual(agent.model_metrics["request_count"], 1)
        self.assertEqual(agent.model_metrics["responses_with_usage"], 1)
        self.assertEqual(agent.model_metrics["usage_collection_errors"], 0)

    def test_agent_retries_without_unsupported_temperature(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="wait"))],
            usage=None,
        )
        completions = TemperatureRejectingCompletions(response)
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        adapter = ChatModelClient(
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            client=client,
        )
        agent = ChatAgent(
            model="no-temperature-model",
            provider="openai-compatible",
            base_url="http://localhost:8000/v1",
            model_client=adapter,
            allow_comm=False,
            log_chat=False,
        )

        self.assertEqual(agent.makeAPIcall(), "wait")
        self.assertEqual(len(completions.requests), 2)
        self.assertIn("temperature", completions.requests[0])
        self.assertNotIn("temperature", completions.requests[1])
        self.assertEqual(agent.model_metrics["failed_requests"], 1)
        self.assertEqual(agent.model_metrics["successful_requests"], 1)


if __name__ == "__main__":
    unittest.main()
