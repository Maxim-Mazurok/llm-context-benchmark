from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import MemorySample, PhaseResult
from .monitor import summarize_phase
from .reporting import (
    analyze,
    write_charts,
    write_dashboard,
    write_records,
    write_report,
)


def _int(row: dict[str, str], key: str) -> int | None:
    value = row.get(key, "")
    try:
        return int(float(value)) if value not in ("", None) else None
    except (TypeError, ValueError):
        return None


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _sample(row: dict[str, str]) -> MemorySample:
    return MemorySample(
        timestamp=row.get("timestamp", ""),
        elapsed_s=_float(row, "elapsed_s"),
        phase=row.get("phase", "idle"),
        phase_id=_int(row, "phase_id") or 0,
        decode_kind=row.get("decode_kind", ""),
        context_tokens=_int(row, "context_tokens") or 0,
        cumulative_input_tokens=_int(row, "cumulative_input_tokens") or 0,
        cumulative_generated_tokens=_int(row, "cumulative_generated_tokens") or 0,
        process_rss_bytes=_int(row, "process_rss_bytes"),
        process_peak_rss_bytes=_int(row, "process_peak_rss_bytes"),
        mlx_active_bytes=_int(row, "mlx_active_bytes"),
        mlx_peak_bytes=_int(row, "mlx_peak_bytes"),
        mlx_cache_bytes=_int(row, "mlx_cache_bytes"),
        system_total_bytes=_int(row, "system_total_bytes"),
        system_used_bytes=_int(row, "system_used_bytes"),
        system_available_bytes=_int(row, "system_available_bytes"),
        system_free_bytes=_int(row, "system_free_bytes"),
        compressed_bytes=_int(row, "compressed_bytes"),
        swap_used_bytes=_int(row, "swap_used_bytes"),
        swapout_bytes=_int(row, "swapout_bytes"),
        pageout_bytes=_int(row, "pageout_bytes"),
        memory_pressure_level=row.get("memory_pressure_level") or None,
        memory_pressure_value=_int(row, "memory_pressure_value"),
    )


def _inferred_model(run_dir: Path) -> str:
    remainder = run_dir.name.split("--", 1)[-1]
    provider, separator, model = remainder.partition("--")
    relative = Path(provider) / model if separator else Path(remainder)
    candidate = Path.home() / ".omlx" / "models" / relative
    return str(candidate if candidate.exists() else relative)


def recover_interrupted_run(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    samples_path = run_dir / "samples.csv"
    if not samples_path.is_file():
        raise ValueError(f"No samples.csv found in {run_dir}")

    groups: list[tuple[list[MemorySample], MemorySample | None]] = []
    current: list[MemorySample] = []
    current_key: tuple[int, str] | None = None
    with samples_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample = _sample(row)
            if sample.phase_id <= 0 or sample.phase not in ("prefill", "decode"):
                continue
            key = (sample.phase_id, sample.phase)
            if current_key is not None and key != current_key:
                groups.append((current, sample))
                current = []
            current_key = key
            current.append(sample)
    if current:
        groups.append((current, None))

    phases: list[PhaseResult] = []
    representative_samples: list[MemorySample] = []
    initial_swap = groups[0][0][0].swap_used_bytes if groups else None
    for samples, next_sample in groups:
        first, last = samples[0], samples[-1]
        representative_samples.append(last)
        if first.phase == "prefill":
            end = next_sample or last
            tokens = max(
                0, end.cumulative_input_tokens - first.cumulative_input_tokens
            )
            duration = max(0.0, end.elapsed_s - first.elapsed_s)
            context_end = end.context_tokens
            cumulative_input = end.cumulative_input_tokens
            cumulative_generated = first.cumulative_generated_tokens
            cache_catchup = 0
        else:
            tokens = max(
                0, last.cumulative_generated_tokens - first.cumulative_generated_tokens
            )
            duration = max(0.0, last.elapsed_s - first.elapsed_s)
            context_end = last.context_tokens
            cumulative_input = last.cumulative_input_tokens
            cumulative_generated = last.cumulative_generated_tokens
            cache_catchup = 0
        status = "completed" if next_sample is not None else "interrupted"
        recovered_phase = summarize_phase(
            phase_id=first.phase_id,
            phase=first.phase,
            decode_kind=first.decode_kind,
            context_start=first.context_tokens,
            context_end=context_end,
            cumulative_input=cumulative_input,
            cumulative_generated=cumulative_generated,
            tokens=tokens,
            cache_catchup_tokens=cache_catchup,
            duration_s=duration,
            tokens_per_second=(tokens / duration if tokens and duration else None),
            samples=samples,
            swap_baseline_bytes=initial_swap,
            status=status,
            error=(
                "Recovered from an unfinalized run"
                if status != "completed"
                else None
            ),
        )
        # Historical builds used the wrong mixed-width Mach VM structure.
        # Swap, pressure, MLX, and psutil values were independent and remain valid.
        recovered_phase.compressed_delta_bytes = None
        recovered_phase.compressed_start_bytes = None
        recovered_phase.compressed_end_bytes = None
        phases.append(recovered_phase)

    try:
        config = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        config = {}
    summary = analyze(
        phases,
        initial_swap=initial_swap,
        practical_decode_ratio=float(config.get("practical_decode_ratio", 0.8)),
        practical_swap_growth_bytes=int(
            float(config.get("practical_swap_growth_gib", 1.0)) * 1024**3
        ),
        no_swap_epsilon_bytes=int(
            float(config.get("no_swap_epsilon_mib", 64.0)) * 1024**2
        ),
        stop_reason="user_interrupt_uncheckpointed",
        hard_limit=None,
    )
    metadata = {
        "model": _inferred_model(run_dir),
        "adapter": "mlx-lm",
        "status": "interrupted",
        "resumable": False,
        "recovered_from_raw_samples": True,
        "measurement_schema_version": 1,
        "output_dir": str(run_dir),
    }
    write_records(run_dir / "phases", phases)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    (run_dir / "run-state.json").write_text(
        json.dumps(
            {
                "status": "interrupted",
                "stop_reason": "user_interrupt_uncheckpointed",
                "resumable": False,
            },
            indent=2,
        )
        + "\n"
    )
    write_charts(run_dir, phases, representative_samples, initial_swap)
    write_dashboard(
        run_dir, phases, representative_samples, summary, metadata, initial_swap
    )
    write_report(run_dir, summary, metadata, phases)
    return summary


def upgrade_legacy_memory_metrics(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.expanduser().resolve()
    try:
        phase_values = json.loads((run_dir / "phases.json").read_text())
        existing_summary = json.loads((run_dir / "summary.json").read_text())
        metadata = json.loads((run_dir / "run-metadata.json").read_text())
        config = json.loads((run_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot upgrade {run_dir}: {exc}") from exc
    phases = [PhaseResult(**value) for value in phase_values]
    by_key = {(phase.phase_id, phase.phase): phase for phase in phases}
    samples_by_key: dict[tuple[int, str], list[MemorySample]] = {}
    with (run_dir / "samples.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            sample = _sample(row)
            key = (sample.phase_id, sample.phase)
            if key in by_key:
                samples_by_key.setdefault(key, []).append(sample)
    initial_swap = existing_summary.get("initial_swap_used_bytes")
    representatives: list[MemorySample] = []
    for key, phase in by_key.items():
        samples = samples_by_key.get(key, [])
        if not samples:
            continue
        representatives.append(samples[-1])
        measured = summarize_phase(
            phase_id=phase.phase_id,
            phase=phase.phase,
            decode_kind=phase.decode_kind,
            context_start=phase.context_start_tokens,
            context_end=phase.context_end_tokens,
            cumulative_input=phase.cumulative_input_tokens,
            cumulative_generated=phase.cumulative_generated_tokens,
            tokens=phase.tokens,
            cache_catchup_tokens=phase.cache_catchup_tokens,
            duration_s=phase.duration_s,
            tokens_per_second=phase.tokens_per_second,
            samples=samples,
            swap_baseline_bytes=initial_swap,
            status=phase.status,
            error=phase.error,
        )
        for field in (
            "rss_mean_bytes",
            "rss_median_bytes",
            "rss_p95_bytes",
            "rss_max_bytes",
            "mlx_active_max_bytes",
            "mlx_peak_max_bytes",
            "mlx_cache_max_bytes",
            "swap_delta_bytes",
            "swap_used_start_bytes",
            "swap_used_end_bytes",
            "swap_used_peak_bytes",
            "swap_growth_peak_bytes",
            "system_available_min_bytes",
            "pressure_worst",
            "pressure_warning_fraction",
            "pressure_critical_fraction",
        ):
            setattr(phase, field, getattr(measured, field))
        phase.compressed_delta_bytes = None
        phase.compressed_start_bytes = None
        phase.compressed_end_bytes = None
        phase.swapout_delta_bytes = None

    summary = analyze(
        phases,
        initial_swap=initial_swap,
        practical_decode_ratio=float(config.get("practical_decode_ratio", 0.8)),
        practical_swap_growth_bytes=int(
            float(config.get("practical_swap_growth_gib", 1.0)) * 1024**3
        ),
        no_swap_epsilon_bytes=int(
            float(config.get("no_swap_epsilon_mib", 64.0)) * 1024**2
        ),
        stop_reason=str(existing_summary.get("stop_reason", "unknown")),
        hard_limit=existing_summary.get("hard_model_runtime_limit_tokens"),
    )
    metadata["measurement_schema_version"] = 1
    metadata["legacy_memory_metrics_upgraded"] = True
    write_records(run_dir / "phases", phases)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (run_dir / "run-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    write_charts(run_dir, phases, representatives, initial_swap)
    write_dashboard(run_dir, phases, representatives, summary, metadata, initial_swap)
    write_report(run_dir, summary, metadata, phases)
    return summary
