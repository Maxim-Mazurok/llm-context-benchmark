from __future__ import annotations

import json
import platform
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from .adapters.base import Adapter
from .models import PhaseResult, TokenTiming
from .monitor import MemoryMonitor, summarize_phase
from .reporting import (
    GIB,
    analyze,
    write_charts,
    write_dashboard,
    write_records,
    write_report,
)


def _gib(value: int | None) -> str:
    return "n/a" if value is None else f"{value / GIB:.2f} GiB"


def _swap_growth(current: int | None, baseline: int | None) -> str:
    if current is None or baseline is None:
        return "n/a"
    return f"{(current - baseline) / GIB:+.2f} GiB"


@dataclass(slots=True)
class BenchmarkConfig:
    chunk_tokens: int = 5_000
    short_decode_tokens: int = 256
    long_decode_tokens: int = 1_000
    long_decode_interval: int = 25_000
    sample_interval_ms: int = 100
    max_context: int | None = None
    swap_stop_gib: float = 4.0
    severe_decode_ratio: float | None = None
    severe_decode_consecutive: int = 2
    practical_decode_ratio: float = 0.8
    practical_swap_growth_gib: float = 1.0
    no_swap_epsilon_mib: float = 64.0
    stop_on_critical_pressure: bool = False


class BenchmarkRunner:
    def __init__(
        self,
        adapter: Adapter,
        config: BenchmarkConfig,
        output_dir: Path,
        *,
        resume: bool = False,
    ) -> None:
        self.adapter = adapter
        self.config = config
        self.output_dir = output_dir
        self.phases: list[PhaseResult] = []
        self.token_timings: list[TokenTiming] = []
        self.context_tokens = 0
        self.input_tokens = 0
        self.generated_tokens = 0
        self._phase_id = 0
        self.sequence_tokens: list[int] = []
        self.resume = resume

    @staticmethod
    def _atomic_json(path: Path, value: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)

    def _load_checkpoint(self) -> None:
        checkpoint_path = self.output_dir / "checkpoint.json"
        if not checkpoint_path.is_file():
            raise ValueError(
                "This run has no resumable checkpoint. It can still be viewed, "
                "but it predates checkpoint support."
            )
        checkpoint = json.loads(checkpoint_path.read_text())
        self.sequence_tokens = [int(token) for token in checkpoint.get("tokens", [])]
        phase_values = json.loads((self.output_dir / "phases.json").read_text())
        self.phases = [PhaseResult(**value) for value in phase_values]
        self._restore_completed_state()
        tokens_path = self.output_dir / "tokens.json"
        if tokens_path.is_file():
            self.token_timings = [
                TokenTiming(**value) for value in json.loads(tokens_path.read_text())
            ]
            self.token_timings = [
                timing
                for timing in self.token_timings
                if timing.phase_id <= self._phase_id
            ]

    def _restore_completed_state(self) -> None:
        if not self.phases:
            self.context_tokens = len(self.sequence_tokens)
            self.input_tokens = 0
            self.generated_tokens = 0
            self._phase_id = 0
            self.token_timings = []
            return
        last = self.phases[-1]
        self.context_tokens = min(last.context_end_tokens, len(self.sequence_tokens))
        self.input_tokens = last.cumulative_input_tokens
        self.generated_tokens = last.cumulative_generated_tokens
        self._phase_id = max(phase.phase_id for phase in self.phases)
        self.token_timings = [
            timing for timing in self.token_timings if timing.phase_id <= self._phase_id
        ]

    def _metadata(self, status: str) -> dict[str, object]:
        return {
            **self.adapter.metadata(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            "updated_at": datetime.now(UTC).isoformat(),
            "output_dir": str(self.output_dir),
            "status": status,
            "resumable": bool(self.sequence_tokens),
            "measurement_schema_version": 2,
        }

    def _write_progress(
        self,
        *,
        initial_swap: int | None,
        hard_limit: int | None,
        stop_reason: str,
        status: str,
    ) -> dict[str, object]:
        summary = analyze(
            self.phases,
            initial_swap=initial_swap,
            practical_decode_ratio=self.config.practical_decode_ratio,
            practical_swap_growth_bytes=int(
                self.config.practical_swap_growth_gib * GIB
            ),
            no_swap_epsilon_bytes=int(self.config.no_swap_epsilon_mib * 1024**2),
            stop_reason=stop_reason,
            hard_limit=hard_limit,
        )
        write_records(self.output_dir / "phases", self.phases)
        self._atomic_json(
            self.output_dir / "checkpoint.json",
            {
                "version": 1,
                "context_tokens": self.context_tokens,
                "input_tokens": self.input_tokens,
                "generated_tokens": self.generated_tokens,
                "phase_id": self._phase_id,
                "tokens": self.sequence_tokens,
            },
        )
        self._atomic_json(self.output_dir / "summary.json", summary)
        self._atomic_json(
            self.output_dir / "run-metadata.json", self._metadata(status)
        )
        self._atomic_json(
            self.output_dir / "run-state.json",
            {
                "status": status,
                "stop_reason": stop_reason,
                "pid": __import__("os").getpid() if status == "running" else None,
                "updated_at": datetime.now(UTC).isoformat(),
                "resumable": bool(self.sequence_tokens),
            },
        )
        return summary

    def _next_phase_id(self) -> int:
        self._phase_id += 1
        return self._phase_id

    def run(self) -> dict[str, object]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.resume:
            self._load_checkpoint()
        (self.output_dir / "config.json").write_text(
            json.dumps(asdict(self.config), indent=2) + "\n"
        )
        monitor = MemoryMonitor(
            self.output_dir,
            interval_s=self.config.sample_interval_ms / 1000,
            mlx_metrics=self.adapter.mlx_metrics,
            append=self.resume,
        )
        stop_reason = "unknown"
        initial_swap: int | None = None
        swap_sample_start = 0
        hard_limit: int | None = None
        baseline_rates: list[float] = []
        collapsed_streak = 0
        monitor.start()
        try:
            self._atomic_json(
                self.output_dir / "run-state.json",
                {
                    "status": "loading" if not self.resume else "restoring",
                    "stop_reason": "running",
                    "pid": __import__("os").getpid(),
                    "updated_at": datetime.now(UTC).isoformat(),
                    "resumable": bool(self.sequence_tokens),
                },
            )
            self._atomic_json(
                self.output_dir / "run-metadata.json",
                self._metadata("restoring" if self.resume else "loading"),
            )
            monitor.update(phase="load")
            self.adapter.load()
            if self.resume and self.sequence_tokens:
                self.adapter.restore_context(self.sequence_tokens)
            hard_limit = self.adapter.context_limit
            effective_limit = self.config.max_context
            if hard_limit is not None:
                effective_limit = (
                    min(effective_limit, hard_limit) if effective_limit else hard_limit
                )
            monitor.update(phase="idle")
            initial = monitor.sample_now()
            initial_swap = initial.swap_used_bytes
            swap_sample_start = max(0, len(monitor.samples) - 1)
            next_long = self.config.long_decode_interval

            while True:
                remaining = (
                    effective_limit - self.context_tokens
                    if effective_limit is not None
                    else None
                )
                if remaining is not None and remaining <= 1:
                    stop_reason = "runtime_context_limit"
                    break
                first_cycle = self.context_tokens == 0
                final_planned_cycle = (
                    remaining is not None
                    and remaining
                    <= self.config.chunk_tokens + 2 * self.config.long_decode_tokens
                )
                milestone_cycle = (
                    self.context_tokens + self.config.chunk_tokens >= next_long
                )
                decode_kind = (
                    "long"
                    if first_cycle or final_planned_cycle or milestone_cycle
                    else "short"
                )
                decode_target = (
                    self.config.long_decode_tokens
                    if decode_kind == "long"
                    else self.config.short_decode_tokens
                )
                input_count = self.config.chunk_tokens
                if remaining is not None:
                    # Preserve at least one newly appended input token. Very
                    # small test caps may therefore use a shortened final
                    # sustained probe; real model limits normally fit it whole.
                    decode_target = min(decode_target, remaining - 1)
                    input_count = (
                        remaining - decode_target
                        if final_planned_cycle
                        else min(input_count, remaining - decode_target)
                    )
                if input_count <= 0:
                    stop_reason = "runtime_context_limit"
                    break

                prefill_id, decode_id = self._next_phase_id(), self._next_phase_id()
                context_before = self.context_tokens
                generated_before = self.generated_tokens
                input_tokens = self.adapter.make_input_tokens(
                    input_count, first=self.context_tokens == 0
                )
                self.adapter.reset_peak_memory()
                monitor.update(
                    phase="prefill",
                    phase_id=prefill_id,
                    decode_kind="",
                    context_tokens=self.context_tokens,
                    cumulative_input_tokens=self.input_tokens,
                    cumulative_generated_tokens=self.generated_tokens,
                )
                monitor.sample_now()
                decode_context_start = context_before + len(input_tokens)
                cycle_input_count = len(input_tokens)
                cycle_decode_id = decode_id
                cycle_decode_kind = decode_kind

                def on_prefill_complete(
                    cycle_input_count: int = cycle_input_count,
                    cycle_decode_id: int = cycle_decode_id,
                    cycle_decode_kind: str = cycle_decode_kind,
                ) -> None:
                    nonlocal decode_context_start
                    monitor.sample_now()
                    self.adapter.reset_peak_memory()
                    self.context_tokens += cycle_input_count
                    self.input_tokens += cycle_input_count
                    decode_context_start = self.context_tokens
                    monitor.update(
                        phase="decode",
                        phase_id=cycle_decode_id,
                        decode_kind=cycle_decode_kind,
                        context_tokens=self.context_tokens,
                        cumulative_input_tokens=self.input_tokens,
                        cumulative_generated_tokens=self.generated_tokens,
                    )
                    monitor.sample_now()

                first_token_time: float | None = None

                def on_token(
                    _token: int,
                    index: int,
                    timestamp: float,
                    cycle_decode_id: int = cycle_decode_id,
                    cycle_decode_kind: str = cycle_decode_kind,
                ) -> None:
                    nonlocal first_token_time
                    previous = (
                        first_token_time
                        if index == 0
                        else self.token_timings[-1].elapsed_s
                    )
                    if index == 0:
                        first_token_time = timestamp
                        interval = None
                    else:
                        interval = timestamp - float(previous)
                    self.context_tokens += 1
                    self.generated_tokens += 1
                    monitor.update(
                        context_tokens=self.context_tokens,
                        cumulative_generated_tokens=self.generated_tokens,
                    )
                    self.token_timings.append(
                        TokenTiming(
                            phase_id=cycle_decode_id,
                            decode_kind=cycle_decode_kind,
                            token_index=index + 1,
                            context_tokens=self.context_tokens,
                            elapsed_s=timestamp,
                            interval_s=interval,
                            instantaneous_tokens_per_second=(
                                1.0 / interval if interval and interval > 0 else None
                            ),
                        )
                    )

                result = self.adapter.append_and_decode(
                    input_tokens, decode_target, on_prefill_complete, on_token
                )
                monitor.sample_now()

                prefill_duration = result.prefill_finished - result.prefill_started
                prefill_samples = monitor.samples_for_phase(prefill_id, "prefill")
                prompt_tokens_processed = (
                    len(input_tokens) + result.cache_catchup_tokens
                )
                self.phases.append(
                    summarize_phase(
                        phase_id=prefill_id,
                        phase="prefill",
                        decode_kind="",
                        context_start=context_before,
                        context_end=decode_context_start,
                        cumulative_input=self.input_tokens,
                        cumulative_generated=generated_before,
                        tokens=len(input_tokens),
                        cache_catchup_tokens=result.cache_catchup_tokens,
                        duration_s=prefill_duration,
                        tokens_per_second=(
                            prompt_tokens_processed / prefill_duration
                            if prefill_duration > 0
                            else None
                        ),
                        samples=prefill_samples,
                        swap_baseline_bytes=initial_swap,
                    )
                )

                if len(result.token_timestamps) >= 2:
                    decode_duration = (
                        result.token_timestamps[-1] - result.token_timestamps[0]
                    )
                    decode_rate = (
                        (len(result.token_timestamps) - 1) / decode_duration
                        if decode_duration > 0
                        else None
                    )
                else:
                    decode_duration, decode_rate = 0.0, None
                decode_samples = monitor.samples_for_phase(decode_id, "decode")
                decode_phase = summarize_phase(
                    phase_id=decode_id,
                    phase="decode",
                    decode_kind=decode_kind,
                    context_start=decode_context_start,
                    context_end=self.context_tokens,
                    cumulative_input=self.input_tokens,
                    cumulative_generated=self.generated_tokens,
                    tokens=len(result.generated_tokens),
                    cache_catchup_tokens=0,
                    duration_s=decode_duration,
                    tokens_per_second=decode_rate,
                    samples=decode_samples,
                    swap_baseline_bytes=initial_swap,
                )
                self.phases.append(decode_phase)
                self.sequence_tokens.extend(input_tokens)
                self.sequence_tokens.extend(result.generated_tokens)
                latest = monitor.sample_now()

                print(
                    f"{self.context_tokens:>8,} ctx | prefill {self.phases[-2].tokens_per_second or 0:>8,.1f} tok/s | "
                    f"{decode_kind:>5} decode {decode_rate or 0:>7,.1f} tok/s\n"
                    f"             MLX {_gib(latest.mlx_active_bytes)} | "
                    f"available {_gib(latest.system_available_bytes)} | "
                    f"compressed {_gib(latest.compressed_bytes)} | "
                    f"swap now {_gib(latest.swap_used_bytes)} | "
                    f"peak Δ {_swap_growth(monitor.peak_swap_used(swap_sample_start), initial_swap)} | "
                    f"pressure {latest.memory_pressure_level or 'n/a'}",
                    flush=True,
                )

                if decode_kind == "long":
                    while next_long <= self.context_tokens:
                        next_long += self.config.long_decode_interval
                self._write_progress(
                    initial_swap=initial_swap,
                    hard_limit=hard_limit,
                    stop_reason="running",
                    status="running",
                )
                if decode_rate:
                    if decode_kind == "short" and len(baseline_rates) < 3:
                        baseline_rates.append(decode_rate)
                    if self.config.severe_decode_ratio is not None:
                        baseline = (
                            statistics.median(baseline_rates)
                            if baseline_rates
                            else decode_rate
                        )
                        collapsed_streak = (
                            collapsed_streak + 1
                            if decode_rate < baseline * self.config.severe_decode_ratio
                            else 0
                        )
                        if collapsed_streak >= self.config.severe_decode_consecutive:
                            stop_reason = "severe_throughput_collapse"
                            break
                if (
                    initial_swap is not None
                    and monitor.peak_swap_used(swap_sample_start) is not None
                    and monitor.peak_swap_used(swap_sample_start) - initial_swap
                    >= self.config.swap_stop_gib * GIB
                ):
                    stop_reason = "swap_threshold"
                    break
                if (
                    self.config.stop_on_critical_pressure
                    and latest.memory_pressure_level == "critical"
                ):
                    stop_reason = "critical_memory_pressure"
                    break
                if (
                    effective_limit is not None
                    and self.context_tokens >= effective_limit
                ):
                    stop_reason = "runtime_context_limit"
                    break
        except KeyboardInterrupt:
            self._restore_completed_state()
            stop_reason = "user_interrupt"
        except Exception as exc:  # noqa: BLE001 - model runtimes use many error types
            self._restore_completed_state()
            message = str(exc)
            allocation_words = (
                "alloc",
                "memory",
                "metal",
                "kv cache",
                "context length",
                "maximum context",
            )
            stop_reason = (
                "allocation_or_context_failure"
                if any(word in message.lower() for word in allocation_words)
                else "runtime_error"
            )
            (self.output_dir / "error.txt").write_text(
                f"{type(exc).__name__}: {message}\n"
            )
        finally:
            monitor.update(phase="idle")
            close_adapter = getattr(self.adapter, "close", None)
            if callable(close_adapter):
                close_adapter()
            monitor.stop()

        status = (
            "interrupted"
            if stop_reason == "user_interrupt"
            else "failed"
            if stop_reason in ("runtime_error", "allocation_or_context_failure")
            else "completed"
        )
        summary = self._write_progress(
            initial_swap=initial_swap,
            hard_limit=hard_limit,
            stop_reason=stop_reason,
            status=status,
        )
        metadata = self._metadata(status)
        write_records(self.output_dir / "tokens", self.token_timings)
        write_charts(self.output_dir, self.phases, monitor.samples, initial_swap)
        write_dashboard(
            self.output_dir,
            self.phases,
            monitor.samples,
            summary,
            metadata,
            initial_swap,
        )
        write_report(self.output_dir, summary, metadata, self.phases)
        return summary
