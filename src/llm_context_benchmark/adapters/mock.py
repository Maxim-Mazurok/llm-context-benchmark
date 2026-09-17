from __future__ import annotations

import time
from collections.abc import Callable

from .base import CycleResult


class MockAdapter:
    """Fast deterministic adapter for smoke tests and report development."""

    name = "mock"

    def __init__(
        self, model: str = "mock", context_limit: int | None = 100_000
    ) -> None:
        self.model_id = model
        self.context_limit = context_limit
        self._pending_token: int | None = None

    def load(self) -> None:
        return None

    def make_input_tokens(self, count: int, first: bool = False) -> list[int]:
        return [1 + (i % 127) for i in range(count)]

    def restore_context(self, tokens: list[int]) -> None:
        self._pending_token = tokens[-1] if tokens else None

    def append_and_decode(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        started = time.perf_counter()
        time.sleep(min(0.03, len(input_tokens) / 500_000))
        finished = time.perf_counter()
        on_prefill_complete()
        generated: list[int] = []
        timestamps: list[float] = []
        for index in range(max_tokens):
            time.sleep(0.0002)
            now = time.perf_counter()
            token = 100 + index % 31
            generated.append(token)
            timestamps.append(now)
            on_token(token, index, now)
        catchup = 1 if self._pending_token is not None else 0
        self._pending_token = generated[-1] if generated else self._pending_token
        return CycleResult(
            len(input_tokens), catchup, generated, started, finished, timestamps
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
        }
