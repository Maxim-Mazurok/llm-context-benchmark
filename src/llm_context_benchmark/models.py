from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class MemorySample:
    timestamp: str
    elapsed_s: float
    phase: str
    phase_id: int
    decode_kind: str
    context_tokens: int
    cumulative_input_tokens: int
    cumulative_generated_tokens: int
    process_rss_bytes: int | None
    process_peak_rss_bytes: int | None
    mlx_active_bytes: int | None
    mlx_peak_bytes: int | None
    mlx_cache_bytes: int | None
    system_total_bytes: int | None
    system_used_bytes: int | None
    system_available_bytes: int | None
    system_free_bytes: int | None
    compressed_bytes: int | None
    swap_used_bytes: int | None
    swapout_bytes: int | None
    pageout_bytes: int | None
    memory_pressure_level: str | None
    memory_pressure_value: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PhaseResult:
    phase_id: int
    phase: str
    decode_kind: str
    context_start_tokens: int
    context_end_tokens: int
    cumulative_input_tokens: int
    cumulative_generated_tokens: int
    tokens: int
    cache_catchup_tokens: int
    duration_s: float
    tokens_per_second: float | None
    rss_mean_bytes: float | None
    rss_median_bytes: float | None
    rss_p95_bytes: float | None
    rss_max_bytes: int | None
    mlx_active_max_bytes: int | None
    mlx_peak_max_bytes: int | None
    mlx_cache_max_bytes: int | None
    compressed_delta_bytes: int | None
    compressed_start_bytes: int | None
    compressed_end_bytes: int | None
    swap_delta_bytes: int | None
    swap_used_start_bytes: int | None
    swap_used_end_bytes: int | None
    pressure_worst: str | None
    status: str = "completed"
    error: str | None = None
    swap_used_peak_bytes: int | None = None
    swap_growth_peak_bytes: int | None = None
    swapout_delta_bytes: int | None = None
    system_available_min_bytes: int | None = None
    pressure_warning_fraction: float | None = None
    pressure_critical_fraction: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenTiming:
    phase_id: int
    decode_kind: str
    token_index: int
    context_tokens: int
    elapsed_s: float
    interval_s: float | None
    instantaneous_tokens_per_second: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MonitorState:
    phase: str = "idle"
    phase_id: int = 0
    decode_kind: str = ""
    context_tokens: int = 0
    cumulative_input_tokens: int = 0
    cumulative_generated_tokens: int = 0
