from __future__ import annotations

import csv
import json
import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path

from .models import MemorySample, MonitorState, PhaseResult
from .native_metrics import make_native_metrics


class MemoryMonitor:
    def __init__(
        self,
        output_dir: Path,
        interval_s: float = 0.1,
        mlx_metrics: Callable[[], tuple[int | None, int | None, int | None]]
        | None = None,
        append: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.interval_s = interval_s
        self._mlx_metrics = mlx_metrics or (lambda: (None, None, None))
        self._native = make_native_metrics()
        self._state = MonitorState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.perf_counter()
        self.samples: list[MemorySample] = []
        self._csv_file = None
        self._jsonl_file = None
        self._writer = None
        self._append = append

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "samples.csv"
        jsonl_path = self.output_dir / "samples.jsonl"
        existing = self._append and csv_path.exists() and csv_path.stat().st_size > 0
        self._csv_file = csv_path.open(
            "a" if self._append else "w", newline="", buffering=1
        )
        self._jsonl_file = jsonl_path.open(
            "a" if self._append else "w", buffering=1
        )
        names = [field.name for field in fields(MemorySample)]
        self._writer = csv.DictWriter(self._csv_file, fieldnames=names)
        if not existing:
            self._writer.writeheader()
        self._started = time.perf_counter()
        self._thread = threading.Thread(
            target=self._run, name="memory-sampler", daemon=True
        )
        self._thread.start()

    def update(self, **changes: object) -> None:
        with self._lock:
            for name, value in changes.items():
                setattr(self._state, name, value)

    def state(self) -> MonitorState:
        with self._lock:
            return MonitorState(
                **{
                    field.name: getattr(self._state, field.name)
                    for field in fields(MonitorState)
                }
            )

    def sample_now(self) -> MemorySample:
        state = self.state()
        native = self._native.snapshot()
        try:
            mlx_active, mlx_peak, mlx_cache = self._mlx_metrics()
        except Exception:  # noqa: BLE001 - optional runtime counters must not stop sampling
            mlx_active = mlx_peak = mlx_cache = None
        sample = MemorySample(
            timestamp=datetime.now(UTC).isoformat(),
            elapsed_s=time.perf_counter() - self._started,
            phase=state.phase,
            phase_id=state.phase_id,
            decode_kind=state.decode_kind,
            context_tokens=state.context_tokens,
            cumulative_input_tokens=state.cumulative_input_tokens,
            cumulative_generated_tokens=state.cumulative_generated_tokens,
            process_rss_bytes=native.process_rss_bytes,
            process_peak_rss_bytes=native.process_peak_rss_bytes,
            mlx_active_bytes=mlx_active,
            mlx_peak_bytes=mlx_peak,
            mlx_cache_bytes=mlx_cache,
            system_total_bytes=native.system_total_bytes,
            system_used_bytes=native.system_used_bytes,
            system_available_bytes=native.system_available_bytes,
            system_free_bytes=native.system_free_bytes,
            compressed_bytes=native.compressed_bytes,
            swap_used_bytes=native.swap_used_bytes,
            swapout_bytes=native.swapout_bytes,
            pageout_bytes=native.pageout_bytes,
            memory_pressure_level=native.memory_pressure_level,
            memory_pressure_value=native.memory_pressure_value,
        )
        with self._lock:
            self.samples.append(sample)
            row = sample.to_dict()
            self._writer.writerow(row)
            self._jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")
        return sample

    def _run(self) -> None:
        deadline = time.perf_counter()
        while not self._stop.is_set():
            self.sample_now()
            deadline += self.interval_s
            self._stop.wait(max(0.0, deadline - time.perf_counter()))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(2.0, self.interval_s * 4))
        self.sample_now()
        if self._csv_file:
            self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()

    def samples_for_phase(self, phase_id: int, phase: str) -> list[MemorySample]:
        with self._lock:
            return [
                s for s in self.samples if s.phase_id == phase_id and s.phase == phase
            ]

    def peak_swap_used(self, start_index: int = 0) -> int | None:
        with self._lock:
            return max(
                (
                    s.swap_used_bytes
                    for s in self.samples[start_index:]
                    if s.swap_used_bytes is not None
                ),
                default=None,
            )


def _percentile(values: list[int], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_phase(
    *,
    phase_id: int,
    phase: str,
    decode_kind: str,
    context_start: int,
    context_end: int,
    cumulative_input: int,
    cumulative_generated: int,
    tokens: int,
    cache_catchup_tokens: int,
    duration_s: float,
    tokens_per_second: float | None,
    samples: list[MemorySample],
    swap_baseline_bytes: int | None = None,
    status: str = "completed",
    error: str | None = None,
) -> PhaseResult:
    rss = [v for s in samples if (v := s.process_rss_bytes) is not None]
    compressed = [v for s in samples if (v := s.compressed_bytes) is not None]
    swap = [v for s in samples if (v := s.swap_used_bytes) is not None]
    swapouts = [v for s in samples if (v := s.swapout_bytes) is not None]
    available = [
        v for s in samples if (v := s.system_available_bytes) is not None
    ]
    pressure_rank = {None: -1, "normal": 0, "warning": 1, "critical": 2}
    pressure = max(
        (s.memory_pressure_level for s in samples),
        key=lambda x: pressure_rank.get(x, -1),
        default=None,
    )
    pressure_count = len(samples)
    warning_count = sum(
        s.memory_pressure_level in ("warning", "critical") for s in samples
    )
    critical_count = sum(s.memory_pressure_level == "critical" for s in samples)
    return PhaseResult(
        phase_id=phase_id,
        phase=phase,
        decode_kind=decode_kind,
        context_start_tokens=context_start,
        context_end_tokens=context_end,
        cumulative_input_tokens=cumulative_input,
        cumulative_generated_tokens=cumulative_generated,
        tokens=tokens,
        cache_catchup_tokens=cache_catchup_tokens,
        duration_s=duration_s,
        tokens_per_second=tokens_per_second,
        rss_mean_bytes=statistics.fmean(rss) if rss else None,
        rss_median_bytes=statistics.median(rss) if rss else None,
        rss_p95_bytes=_percentile(rss, 0.95),
        rss_max_bytes=max(rss) if rss else None,
        mlx_active_max_bytes=max(
            (s.mlx_active_bytes for s in samples if s.mlx_active_bytes is not None),
            default=None,
        ),
        mlx_peak_max_bytes=max(
            (s.mlx_peak_bytes for s in samples if s.mlx_peak_bytes is not None),
            default=None,
        ),
        mlx_cache_max_bytes=max(
            (s.mlx_cache_bytes for s in samples if s.mlx_cache_bytes is not None),
            default=None,
        ),
        compressed_delta_bytes=(compressed[-1] - compressed[0])
        if len(compressed) >= 2
        else None,
        compressed_start_bytes=compressed[0] if compressed else None,
        compressed_end_bytes=compressed[-1] if compressed else None,
        swap_delta_bytes=(swap[-1] - swap[0]) if len(swap) >= 2 else None,
        swap_used_start_bytes=swap[0] if swap else None,
        swap_used_end_bytes=swap[-1] if swap else None,
        pressure_worst=pressure,
        status=status,
        error=error,
        swap_used_peak_bytes=max(swap) if swap else None,
        swap_growth_peak_bytes=(
            max(swap) - (
                swap_baseline_bytes if swap_baseline_bytes is not None else swap[0]
            )
        )
        if swap
        else None,
        swapout_delta_bytes=(swapouts[-1] - swapouts[0])
        if len(swapouts) >= 2
        else None,
        system_available_min_bytes=min(available) if available else None,
        pressure_warning_fraction=(warning_count / pressure_count)
        if pressure_count
        else None,
        pressure_critical_fraction=(critical_count / pressure_count)
        if pressure_count
        else None,
    )
