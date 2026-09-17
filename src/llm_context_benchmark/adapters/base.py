from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(slots=True)
class CycleResult:
    appended_input_tokens: int
    cache_catchup_tokens: int
    generated_tokens: list[int]
    prefill_started: float
    prefill_finished: float
    token_timestamps: list[float]


class Adapter(Protocol):
    name: str
    model_id: str
    context_limit: int | None

    def load(self) -> None: ...

    def make_input_tokens(self, count: int, first: bool = False) -> list[int]: ...

    def restore_context(self, tokens: list[int]) -> None: ...

    def append_and_decode(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult: ...

    def mlx_metrics(self) -> tuple[int | None, int | None, int | None]: ...

    def reset_peak_memory(self) -> None: ...

    def metadata(self) -> dict[str, object]: ...
