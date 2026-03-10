"""Integration tests for Ollama client. Requires Ollama running locally with mistral:7b pulled."""

import pytest

from phishagent.llm_client import OllamaClient, LLMConnectionError


@pytest.mark.integration
class TestOllamaClientIntegration:
    @pytest.fixture
    def client(self):
        return OllamaClient()

    def test_is_available(self, client):
        assert client.is_available() is True

    def test_generate_returns_content(self, client):
        response = client.generate("mistral:7b", "Say hello in one word.")
        assert len(response.content) > 0
        assert response.token_count > 0

    def test_chat_multi_turn(self, client):
        messages = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"},
        ]
        response = client.chat("mistral:7b", messages)
        assert "alice" in response.content.lower()

    def test_list_models(self, client):
        models = client.list_models()
        assert isinstance(models, list)
        assert len(models) > 0

    def test_ensure_model_valid(self, client):
        assert client.ensure_model("mistral") is True

    def test_ensure_model_invalid(self, client):
        assert client.ensure_model("nonexistent_model_xyz") is False

    def test_timeout_mechanism(self):
        """Verify timeout parameter is accepted (may not actually timeout)."""
        client = OllamaClient(timeout=1)
        # Just verify the client was created with the timeout
        assert client.timeout == 1

    def test_invalid_model_raises(self, client):
        with pytest.raises(Exception):
            client.generate("nonexistent:model", "Hello")

    def test_generate_with_system_prompt(self, client):
        response = client.generate(
            "mistral:7b",
            "What color is the sky?",
            system="You must answer every question with exactly one word.",
        )
        assert len(response.content) > 0
