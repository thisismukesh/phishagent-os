"""Ollama HTTP client. All communication with the locally-running Ollama instance.

No other module touches the network.
"""

import os
import subprocess
import time
from typing import Optional

import httpx

from phishagent.models import LLMResponse
from phishagent.utils import get_logger

logger = get_logger(__name__)


def detect_cuda() -> dict:
    """Detect CUDA GPU availability by querying nvidia-smi.

    Returns a dict with keys: available (bool), count (int), devices (list[str]).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            devices = [line.strip() for line in result.stdout.strip().splitlines() if line.strip()]
            return {"available": True, "count": len(devices), "devices": devices}
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass

    # Fallback: check for NVIDIA device files
    if os.path.exists("/dev/nvidia0"):
        return {"available": True, "count": 1, "devices": ["NVIDIA GPU"]}

    return {"available": False, "count": 0, "devices": []}


# ── Error Hierarchy ─────────────────────────────────────────────────────────────


class LLMClientError(Exception):
    pass


class LLMConnectionError(LLMClientError):
    pass


class LLMModelNotFoundError(LLMClientError):
    pass


class LLMTimeoutError(LLMClientError):
    pass


class LLMResponseError(LLMClientError):
    pass


# ── Client ──────────────────────────────────────────────────────────────────────

# Retry configuration
_MAX_RETRIES = 3
_BACKOFF_SECONDS = [1, 2, 4]


class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
        num_gpu: Optional[int] = None,
    ):
        """Initialize client.

        Args:
            base_url: Ollama server URL.
            timeout: Request timeout in seconds.
            num_gpu: GPU layers to offload. None = auto-detect (uses all GPU layers when
                CUDA is available, CPU-only otherwise). 0 = force CPU. Positive int = N layers.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

        self.cuda_info = detect_cuda()

        if num_gpu is None:
            # Auto: offload all layers to GPU when CUDA is available
            self._num_gpu: Optional[int] = 999 if self.cuda_info["available"] else None
        else:
            self._num_gpu = num_gpu if num_gpu >= 0 else None  # negative = let Ollama decide

        if self.cuda_info["available"]:
            logger.info(
                f"CUDA detected: {self.cuda_info['count']} device(s) — "
                f"{', '.join(self.cuda_info['devices'])} | num_gpu={self._num_gpu}"
            )
        else:
            logger.info("No CUDA GPU detected — running in CPU-only mode")

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Send a single completion request via /api/generate."""
        options: dict = {"temperature": temperature, "num_predict": max_tokens}
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            payload["system"] = system

        data = self._request_with_retry("POST", "/api/generate", payload)

        return LLMResponse(
            content=data.get("response") or "",
            model=data.get("model") or model,
            token_count=(data.get("eval_count") or 0) + (data.get("prompt_eval_count") or 0),
            duration_ms=(data.get("total_duration") or 0) // 1_000_000,  # nanoseconds → ms
            done=data.get("done", True),
        )

    def chat(
        self,
        model: str,
        messages: list[dict],
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> LLMResponse:
        """Send a multi-turn chat request via /api/chat."""
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend(messages)

        options: dict = {"temperature": temperature, "num_predict": max_tokens}
        if self._num_gpu is not None:
            options["num_gpu"] = self._num_gpu

        payload = {
            "model": model,
            "messages": chat_messages,
            "stream": False,
            "options": options,
        }

        data = self._request_with_retry("POST", "/api/chat", payload)

        message_content = (data.get("message") or {}).get("content") or ""

        return LLMResponse(
            content=message_content,
            model=data.get("model") or model,
            token_count=(data.get("eval_count") or 0) + (data.get("prompt_eval_count") or 0),
            duration_ms=(data.get("total_duration") or 0) // 1_000_000,
            done=data.get("done", True),
        )

    def is_available(self) -> bool:
        """Check if Ollama is running and responsive."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def list_models(self) -> list[str]:
        """Return list of locally available model names."""
        try:
            resp = self._client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def ensure_model(self, model: str) -> bool:
        """Check if model is available locally. Return False if not pulled."""
        models = self.list_models()
        # Check exact match or prefix match (e.g., "mistral:7b" matches "mistral:7b-instruct-...")
        return any(model == m or m.startswith(model.split(":")[0] + ":") for m in models)

    def _request_with_retry(self, method: str, path: str, payload: dict) -> dict:
        """Execute an HTTP request with retry logic.

        3 retries with exponential backoff (1s, 2s, 4s) on connection errors and 5xx.
        No retry on 4xx.
        """
        url = f"{self.base_url}{path}"
        last_error = None

        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    f"Request attempt {attempt + 1}/{_MAX_RETRIES}: {method} {path} "
                    f"model={payload.get('model', 'unknown')}"
                )

                resp = self._client.request(method, url, json=payload)

                if resp.status_code == 404:
                    raise LLMModelNotFoundError(
                        f"Model not found: {payload.get('model', 'unknown')}"
                    )
                if 400 <= resp.status_code < 500:
                    raise LLMResponseError(
                        f"Client error {resp.status_code}: {resp.text[:200]}"
                    )
                if resp.status_code >= 500:
                    last_error = LLMResponseError(f"Server error {resp.status_code}")
                    if attempt < _MAX_RETRIES - 1:
                        time.sleep(_BACKOFF_SECONDS[attempt])
                        continue
                    raise last_error

                resp.raise_for_status()
                return resp.json()

            except httpx.ConnectError as e:
                last_error = LLMConnectionError(f"Cannot connect to Ollama at {self.base_url}: {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                raise last_error

            except httpx.TimeoutException as e:
                last_error = LLMTimeoutError(f"Request timed out after {self.timeout}s: {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                raise last_error

            except (LLMModelNotFoundError, LLMResponseError):
                raise

            except Exception as e:
                last_error = LLMClientError(f"Unexpected error: {e}")
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                raise last_error

        raise last_error  # type: ignore[misc]
