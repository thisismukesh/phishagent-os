"""Unit tests for llm_client.py — mock httpx to test all code paths."""

import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from phishagent.llm_client import (
    OllamaClient,
    LLMClientError,
    LLMConnectionError,
    LLMModelNotFoundError,
    LLMResponseError,
    LLMTimeoutError,
    _MAX_RETRIES,
)


class FakeResponse:
    """Fake httpx response."""

    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text or json.dumps(self._data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


class TestOllamaClientInit:
    def test_strips_trailing_slash(self):
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"

    def test_default_timeout(self):
        client = OllamaClient()
        assert client.timeout == 120

    def test_custom_timeout(self):
        client = OllamaClient(timeout=30)
        assert client.timeout == 30


class TestGenerate:
    def test_generate_success(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {
            "response": "Hello world",
            "model": "mistral:7b",
            "eval_count": 10,
            "prompt_eval_count": 5,
            "total_duration": 500_000_000,  # 500ms in nanoseconds
            "done": True,
        })
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        result = client.generate("mistral:7b", "Say hello")
        assert result.content == "Hello world"
        assert result.model == "mistral:7b"
        assert result.token_count == 15
        assert result.duration_ms == 500
        assert result.done is True

    def test_generate_with_system_prompt(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {"response": "Blue", "model": "mistral:7b", "done": True})
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        result = client.generate("mistral:7b", "What color?", system="Answer in one word")
        assert result.content == "Blue"

        # Verify system was included in payload
        call_args = client._client.request.call_args
        payload = call_args[1]["json"]
        assert payload["system"] == "Answer in one word"

    def test_generate_without_system_prompt(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {"response": "Hi", "model": "m", "done": True})
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        client.generate("m", "Hello")
        payload = client._client.request.call_args[1]["json"]
        assert "system" not in payload


class TestChat:
    def test_chat_success(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {
            "message": {"role": "assistant", "content": "I'm doing well!"},
            "model": "mistral:7b",
            "eval_count": 8,
            "prompt_eval_count": 12,
            "total_duration": 1_000_000_000,
            "done": True,
        })
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        messages = [{"role": "user", "content": "How are you?"}]
        result = client.chat("mistral:7b", messages)
        assert result.content == "I'm doing well!"
        assert result.token_count == 20

    def test_chat_with_system(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {
            "message": {"content": "Yo!"},
            "model": "m",
            "done": True,
        })
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        client.chat("m", [{"role": "user", "content": "Hi"}], system="Be casual")
        payload = client._client.request.call_args[1]["json"]
        # System message should be first in the messages list
        assert payload["messages"][0]["role"] == "system"
        assert payload["messages"][0]["content"] == "Be casual"

    def test_chat_without_system(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {"message": {"content": "Ok"}, "model": "m", "done": True})
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        client.chat("m", [{"role": "user", "content": "Hi"}])
        payload = client._client.request.call_args[1]["json"]
        assert payload["messages"][0]["role"] == "user"

    def test_chat_empty_message_content(self):
        client = OllamaClient()
        fake_resp = FakeResponse(200, {"message": {}, "model": "m", "done": True})
        client._client = MagicMock()
        client._client.request.return_value = fake_resp

        result = client.chat("m", [{"role": "user", "content": "Hi"}])
        assert result.content == ""


class TestIsAvailable:
    def test_available(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {"models": []})
        assert client.is_available() is True

    def test_unavailable_connect_error(self):
        import httpx
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.side_effect = httpx.ConnectError("refused")
        assert client.is_available() is False

    def test_unavailable_timeout(self):
        import httpx
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.side_effect = httpx.TimeoutException("timeout")
        assert client.is_available() is False


class TestListModels:
    def test_list_models_success(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {
            "models": [{"name": "mistral:7b"}, {"name": "llama3:8b"}]
        })
        models = client.list_models()
        assert models == ["mistral:7b", "llama3:8b"]

    def test_list_models_empty(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {"models": []})
        assert client.list_models() == []

    def test_list_models_error(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.side_effect = Exception("network error")
        assert client.list_models() == []


class TestEnsureModel:
    def test_exact_match(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {
            "models": [{"name": "mistral:7b"}]
        })
        assert client.ensure_model("mistral:7b") is True

    def test_prefix_match(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {
            "models": [{"name": "mistral:7b-instruct-v0.3"}]
        })
        assert client.ensure_model("mistral:7b") is True

    def test_no_match(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.get.return_value = FakeResponse(200, {
            "models": [{"name": "llama3:8b"}]
        })
        assert client.ensure_model("mistral:7b") is False


class TestRetryLogic:
    @patch("phishagent.llm_client.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        import httpx
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.side_effect = httpx.ConnectError("refused")

        with pytest.raises(LLMConnectionError):
            client.generate("m", "test")

        assert client._client.request.call_count == _MAX_RETRIES
        assert mock_sleep.call_count == _MAX_RETRIES - 1

    @patch("phishagent.llm_client.time.sleep")
    def test_retries_on_timeout(self, mock_sleep):
        import httpx
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.side_effect = httpx.TimeoutException("timeout")

        with pytest.raises(LLMTimeoutError):
            client.generate("m", "test")

        assert client._client.request.call_count == _MAX_RETRIES

    @patch("phishagent.llm_client.time.sleep")
    def test_retries_on_5xx(self, mock_sleep):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.return_value = FakeResponse(500, text="Internal error")

        with pytest.raises(LLMResponseError):
            client.generate("m", "test")

        assert client._client.request.call_count == _MAX_RETRIES

    def test_no_retry_on_404(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.return_value = FakeResponse(404, text="Not found")

        with pytest.raises(LLMModelNotFoundError):
            client.generate("nonexistent:model", "test")

        assert client._client.request.call_count == 1

    def test_no_retry_on_4xx(self):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.return_value = FakeResponse(400, text="Bad request")

        with pytest.raises(LLMResponseError):
            client.generate("m", "test")

        assert client._client.request.call_count == 1

    @patch("phishagent.llm_client.time.sleep")
    def test_retries_on_generic_exception(self, mock_sleep):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.side_effect = RuntimeError("weird error")

        with pytest.raises(LLMClientError):
            client.generate("m", "test")

        assert client._client.request.call_count == _MAX_RETRIES

    @patch("phishagent.llm_client.time.sleep")
    def test_succeeds_after_retry(self, mock_sleep):
        client = OllamaClient()
        client._client = MagicMock()
        import httpx
        # First call fails, second succeeds
        client._client.request.side_effect = [
            httpx.ConnectError("refused"),
            FakeResponse(200, {"response": "OK", "model": "m", "done": True}),
        ]

        result = client.generate("m", "test")
        assert result.content == "OK"
        assert client._client.request.call_count == 2

    @patch("phishagent.llm_client.time.sleep")
    def test_5xx_then_success(self, mock_sleep):
        client = OllamaClient()
        client._client = MagicMock()
        client._client.request.side_effect = [
            FakeResponse(503, text="Service unavailable"),
            FakeResponse(200, {"response": "OK", "model": "m", "done": True}),
        ]

        result = client.generate("m", "test")
        assert result.content == "OK"
