from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import CycleResult


class LlamaServerAdapter:
    """Benchmark one persistent llama.cpp server slot through its HTTP API."""

    name = "llama-server"

    def __init__(
        self,
        model: str | None,
        *,
        server_url: str = "http://127.0.0.1:8080",
        seed_text: str | None = None,
    ) -> None:
        self.model_id = model or ""
        self.server_url = server_url.rstrip("/")
        self.seed_text = seed_text or (
            "The benchmark extends one continuous conversation with neutral prose, "
            "measurements, explanations, and repeated factual structure. "
        )
        self.context_limit: int | None = None
        self._seed_tokens: list[int] = []
        self._sequence_tokens: list[int] = []
        self._api_key = os.environ.get("LLAMA_API_KEY")

    def _request(
        self,
        endpoint: str,
        payload: dict[str, object] | None = None,
        *,
        stream: bool = False,
    ) -> Any:
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = Request(
            f"{self.server_url}{endpoint}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )
        try:
            return urlopen(request, timeout=3_600)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"llama-server returned HTTP {error.code} for {endpoint}: {detail}"
            ) from error
        except URLError as error:
            raise RuntimeError(
                f"Cannot reach llama-server at {self.server_url}: {error.reason}"
            ) from error

    def _json_request(
        self, endpoint: str, payload: dict[str, object] | None = None
    ) -> dict[str, Any]:
        with self._request(endpoint, payload) as response:
            value = json.load(response)
        if not isinstance(value, dict):
            raise RuntimeError(f"llama-server returned invalid JSON for {endpoint}")
        return value

    def load(self) -> None:
        health = self._json_request("/health")
        if health.get("status") != "ok":
            raise RuntimeError(f"llama-server is not ready: {health}")
        models = self._json_request("/v1/models").get("data")
        if not isinstance(models, list) or not models:
            raise RuntimeError("llama-server reported no loaded model")
        model = models[0]
        if not isinstance(model, dict):
            raise RuntimeError("llama-server returned invalid model metadata")
        loaded_model_id = model.get("id")
        if not self.model_id and isinstance(loaded_model_id, str):
            self.model_id = loaded_model_id
        props = self._json_request("/props")
        settings = props.get("default_generation_settings")
        if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
            self.context_limit = settings["n_ctx"]
        tokenized = self._json_request(
            "/tokenize", {"content": self.seed_text, "add_special": False}
        ).get("tokens")
        if not isinstance(tokenized, list) or not tokenized:
            raise RuntimeError("llama-server tokenizer produced no benchmark seed tokens")
        self._seed_tokens = [int(token) for token in tokenized]

    def make_input_tokens(self, count: int, first: bool = False) -> list[int]:
        if not self._seed_tokens:
            raise RuntimeError("llama-server adapter is not loaded")
        prefix: list[int] = []
        if first:
            tokenized = self._json_request(
                "/tokenize", {"content": self.seed_text, "add_special": True}
            ).get("tokens")
            if isinstance(tokenized, list):
                special_count = max(0, len(tokenized) - len(self._seed_tokens))
                prefix = [int(token) for token in tokenized[:special_count]]
        repeated = self._seed_tokens * (
            (max(0, count - len(prefix)) + len(self._seed_tokens) - 1)
            // len(self._seed_tokens)
        )
        return (prefix + repeated)[:count]

    def restore_context(self, tokens: list[int]) -> None:
        if not tokens:
            return
        result = self._json_request(
            "/completion",
            {
                "prompt": tokens,
                "n_predict": 0,
                "cache_prompt": True,
                "id_slot": 0,
            },
        )
        if result.get("truncated"):
            raise RuntimeError("llama-server truncated restored benchmark context")
        self._sequence_tokens = list(tokens)

    def append_and_decode(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        prompt_tokens = [*self._sequence_tokens, *input_tokens]
        started = time.perf_counter()
        response = self._request(
            "/completion",
            {
                "prompt": prompt_tokens,
                "n_predict": max_tokens,
                "cache_prompt": True,
                "return_tokens": True,
                "stream": True,
                "id_slot": 0,
                "temperature": 0,
            },
            stream=True,
        )
        generated_tokens: list[int] = []
        token_timestamps: list[float] = []
        prefill_finished: float | None = None
        cache_catchup_tokens = 0
        with response:
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    continue
                event = json.loads(data)
                if event.get("truncated"):
                    raise RuntimeError("llama-server truncated benchmark context")
                timings = event.get("timings")
                if isinstance(timings, dict) and isinstance(
                    timings.get("cache_n"), int
                ):
                    cache_catchup_tokens = max(
                        0, len(self._sequence_tokens) - timings["cache_n"]
                    )
                event_tokens = event.get("tokens")
                if not isinstance(event_tokens, list) or not event_tokens:
                    continue
                if prefill_finished is None:
                    prefill_finished = time.perf_counter()
                    on_prefill_complete()
                for token_value in event_tokens:
                    token = int(token_value)
                    timestamp = time.perf_counter()
                    generated_tokens.append(token)
                    token_timestamps.append(timestamp)
                    on_token(token, len(generated_tokens) - 1, timestamp)
        if prefill_finished is None:
            prefill_finished = time.perf_counter()
            on_prefill_complete()
        self._sequence_tokens = [*prompt_tokens, *generated_tokens]
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=cache_catchup_tokens,
            generated_tokens=generated_tokens,
            prefill_started=started,
            prefill_finished=prefill_finished,
            token_timestamps=token_timestamps,
        )

    def mlx_metrics(self) -> tuple[int | None, int | None, int | None]:
        return None, None, None

    def reset_peak_memory(self) -> None:
        return None

    def metadata(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model": self.model_id,
            "context_limit": self.context_limit,
            "continuous_cache": True,
            "server_url": self.server_url,
        }