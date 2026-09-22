from __future__ import annotations

import atexit
import json
import os
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .base import CycleResult


class UnslothLlamaCppAdapter:
    """Persistent-cache adapter using Unsloth's bundled llama-server."""

    name = "unsloth-llama.cpp"

    def __init__(
        self,
        model: str,
        *,
        context_size: int | None = None,
        server_path: str | None = None,
        seed_text: str | None = None,
    ) -> None:
        self.model_id = model
        self.context_limit = context_size
        self.context_size = context_size
        self.server_path = server_path or str(
            Path.home() / ".unsloth" / "llama.cpp" / "build" / "bin" / "llama-server"
        )
        self.seed_text = seed_text or (
            "The benchmark extends one continuous conversation with neutral prose, "
            "measurements, explanations, and repeated factual structure. "
        )
        self._seed_tokens: list[int] = []
        self._sequence_tokens: list[int] = []
        self._process: subprocess.Popen[str] | None = None
        self._output_lines: deque[str] = deque(maxlen=40)
        self._output_thread: threading.Thread | None = None
        self._endpoint = ""
        self._runtime_version: str | None = None
        atexit.register(self.close)

    @staticmethod
    def _available_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as network_socket:
            network_socket.bind(("127.0.0.1", 0))
            return int(network_socket.getsockname()[1])

    def _request_json(
        self,
        route: str,
        payload: dict[str, object] | None = None,
        *,
        timeout: float = 10,
    ) -> dict[str, object] | list[object]:
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(
            f"{self._endpoint}{route}",
            data=body,
            headers={"content-type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"Unsloth llama-server returned HTTP {error.code}: {detail}"
            ) from error

    def _capture_output(self) -> None:
        if self._process is None or self._process.stdout is None:
            return
        for line in self._process.stdout:
            self._output_lines.append(line.rstrip())

    def load(self) -> None:
        model_path = Path(self.model_id).expanduser().resolve()
        server_path = Path(self.server_path).expanduser().resolve()
        if not model_path.is_file():
            raise ValueError(f"Unsloth GGUF model does not exist: {model_path}")
        if not os.access(server_path, os.X_OK):
            raise ValueError(f"Unsloth llama-server is not executable: {server_path}")

        port = self._available_port()
        self._endpoint = f"http://127.0.0.1:{port}"
        command = [
            str(server_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--model",
            str(model_path),
            "--parallel",
            "1",
            "--slots",
            "--cache-prompt",
        ]
        if self.context_size is not None:
            command.extend(["--ctx-size", str(self.context_size)])
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._output_thread = threading.Thread(
            target=self._capture_output,
            name="unsloth-llama-server-output",
            daemon=True,
        )
        self._output_thread.start()

        deadline = time.monotonic() + 5 * 60
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                detail = "\n".join(self._output_lines)
                raise RuntimeError(f"Unsloth llama-server exited during load.\n{detail}")
            try:
                health = self._request_json("/health", timeout=1)
                if isinstance(health, dict) and health.get("status") == "ok":
                    break
            except (HTTPError, URLError, TimeoutError, RuntimeError):
                pass
            time.sleep(0.1)
        else:
            raise RuntimeError("Timed out waiting for Unsloth llama-server to load")

        properties = self._request_json("/props")
        if isinstance(properties, dict):
            settings = properties.get("default_generation_settings")
            if isinstance(settings, dict) and isinstance(settings.get("n_ctx"), int):
                self.context_limit = int(settings["n_ctx"])
        try:
            version = self._request_json("/version")
            if isinstance(version, dict) and isinstance(version.get("version"), str):
                self._runtime_version = version["version"]
        except RuntimeError:
            pass
        tokenized = self._request_json("/tokenize", {"content": self.seed_text})
        if not isinstance(tokenized, dict) or not isinstance(tokenized.get("tokens"), list):
            raise RuntimeError("Unsloth llama-server returned invalid tokenizer output")
        self._seed_tokens = [int(token) for token in tokenized["tokens"]]
        if not self._seed_tokens:
            raise RuntimeError("Unsloth tokenizer produced no benchmark seed tokens")

    def make_input_tokens(self, count: int, first: bool = False) -> list[int]:
        if count <= 0:
            return []
        repeats = (count + len(self._seed_tokens) - 1) // len(self._seed_tokens)
        return (self._seed_tokens * repeats)[:count]

    def restore_context(self, tokens: list[int]) -> None:
        self._sequence_tokens = list(tokens)

    def append_and_decode(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        prompt = [*self._sequence_tokens, *input_tokens]
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "cache_prompt": True,
            "id_slot": 0,
            "n_probs": 1,
            "stream": True,
            "timings_per_token": True,
        }
        request = Request(
            f"{self._endpoint}/completion",
            data=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        prefill_started = time.perf_counter()
        prefill_finished: float | None = None
        generated_tokens: list[int] = []
        token_timestamps: list[float] = []
        final_event: dict[str, object] = {}
        try:
            with urlopen(request, timeout=24 * 60 * 60) as response:
                for raw_line in response:
                    line = raw_line.decode().strip()
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("stop"):
                        final_event = event
                        continue
                    event_tokens = event.get("tokens")
                    if not isinstance(event_tokens, list):
                        continue
                    if prefill_finished is None:
                        prefill_finished = time.perf_counter()
                        on_prefill_complete()
                    for token in event_tokens:
                        timestamp = time.perf_counter()
                        token_value = int(token)
                        token_index = len(generated_tokens)
                        generated_tokens.append(token_value)
                        token_timestamps.append(timestamp)
                        on_token(token_value, token_index, timestamp)
        except HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(
                f"Unsloth llama-server completion failed with HTTP {error.code}: {detail}"
            ) from error
        if prefill_finished is None:
            prefill_finished = time.perf_counter()
            on_prefill_complete()
        if not final_event:
            raise RuntimeError("Unsloth llama-server ended without a final event")

        timings = final_event.get("timings")
        prompt_tokens_processed = (
            int(timings.get("prompt_n", len(input_tokens)))
            if isinstance(timings, dict)
            else len(input_tokens)
        )
        prompt_milliseconds = (
            float(timings.get("prompt_ms", 0))
            if isinstance(timings, dict)
            else 0
        )
        if prompt_milliseconds > 0:
            prefill_finished = prefill_started + prompt_milliseconds / 1_000
        cache_catchup_tokens = max(0, prompt_tokens_processed - len(input_tokens))
        self._sequence_tokens.extend(input_tokens)
        self._sequence_tokens.extend(generated_tokens)
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=cache_catchup_tokens,
            generated_tokens=generated_tokens,
            prefill_started=prefill_started,
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
            "runtime": "Unsloth llama.cpp",
            "runtime_version": self._runtime_version,
            "cache_slot": 0,
            "prefill_timing_source": "llama.cpp timings.prompt_ms",
        }

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
