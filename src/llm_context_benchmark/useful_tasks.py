from __future__ import annotations

import csv
import html
import json
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class UsefulTaskObservation:
    schema_version: int
    source_run_id: str
    source_session_id: str
    benchmark: str
    model: str
    task_id: str
    attempt_id: str
    pass_number: int | None
    passed: bool | None
    score: float | None
    request_status: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cached_prompt_tokens: int | None
    request_duration_seconds: float | None
    total_time_seconds: float | None
    time_to_first_token_seconds: float | None
    prompt_eval_duration_seconds: float | None
    generation_duration_seconds: float | None
    prompt_tokens_per_second: float | None
    generation_tokens_per_second: float | None
    active_duration_milliseconds: float | None
    physical_footprint_peak_bytes: int | None
    mlx_active_peak_bytes: int | None
    mlx_cache_peak_bytes: int | None
    system_used_peak_bytes: int | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _number(value: Any, conversion: type[int] | type[float]) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return conversion(value)
    except (TypeError, ValueError):
        return None


def _peak(samples: list[dict[str, Any]], field_name: str) -> int | None:
    values = [
        value
        for sample in samples
        if (value := _number(sample.get(field_name), int)) is not None
    ]
    return max(values) if values else None


def _error_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True)


def _read_json_file(run_directory: Path, file_name: str) -> Any:
    file_path = run_directory / file_name
    try:
        return json.loads(file_path.read_text())
    except FileNotFoundError as error:
        raise ValueError(f"Workbench run is missing {file_name}: {run_directory}") from error
    except OSError as error:
        raise ValueError(f"Cannot read {file_name} from {run_directory}: {error}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {file_name} from {run_directory}: {error}") from error


def load_workbench_observations(
    run_directory: Path,
) -> list[UsefulTaskObservation]:
    run_directory = run_directory.expanduser().resolve()
    run = _read_json_file(run_directory, "run.json")
    results = _read_json_file(run_directory, "results.json")
    telemetry = _read_json_file(run_directory, "telemetry.json")

    if telemetry.get("schemaVersion") != 1:
        raise ValueError("Unsupported Workbench telemetry schema version")
    results_by_attempt = {
        str(result.get("attemptId")): result
        for result in results
        if result.get("attemptId")
    }
    requests: list[tuple[str, dict[str, Any]]] = []
    seen_request_ids: set[str] = set()
    for session in telemetry.get("sessions", []):
        session_id = str(session.get("session_id", ""))
        for request in session.get("requests", []):
            request_id = str(request.get("request_id", ""))
            if not request_id:
                raise ValueError("Telemetry request is missing request_id")
            if request_id in seen_request_ids:
                raise ValueError(f"Duplicate telemetry request_id: {request_id}")
            seen_request_ids.add(request_id)
            requests.append((session_id, request))

    observations: list[UsefulTaskObservation] = []
    for session_id, request in requests:
        request_id = str(request["request_id"])
        result = results_by_attempt.get(request_id, {})
        usage = request.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        host_samples = request.get("host_samples") or []
        prompt_tokens = _number(usage.get("prompt_tokens"), int)
        completion_tokens = _number(usage.get("completion_tokens"), int)
        total_tokens = _number(usage.get("total_tokens"), int)
        if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
            total_tokens = prompt_tokens + completion_tokens
        passed = result.get("passed")
        observations.append(
            UsefulTaskObservation(
                schema_version=1,
                source_run_id=str(telemetry.get("runId") or run.get("id") or ""),
                source_session_id=session_id,
                benchmark=str(telemetry.get("benchmark") or run.get("benchmark") or ""),
                model=str(telemetry.get("model") or run.get("model") or request.get("model_id") or ""),
                task_id=str(result.get("taskId") or request_id.split("::pass-", 1)[0]),
                attempt_id=request_id,
                pass_number=_number(result.get("passNumber"), int),
                passed=passed if isinstance(passed, bool) else None,
                score=_number(result.get("score"), float),
                request_status=str(request.get("status", "unknown")),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cached_prompt_tokens=_number(prompt_details.get("cached_tokens"), int),
                request_duration_seconds=_number(request.get("duration_seconds"), float),
                total_time_seconds=_number(usage.get("total_time"), float),
                time_to_first_token_seconds=_number(usage.get("time_to_first_token"), float),
                prompt_eval_duration_seconds=_number(usage.get("prompt_eval_duration"), float),
                generation_duration_seconds=_number(usage.get("generation_duration"), float),
                prompt_tokens_per_second=_number(usage.get("prompt_tokens_per_second"), float),
                generation_tokens_per_second=_number(usage.get("generation_tokens_per_second"), float),
                active_duration_milliseconds=_number(result.get("activeDurationMilliseconds"), float),
                physical_footprint_peak_bytes=_peak(host_samples, "physical_footprint_bytes"),
                mlx_active_peak_bytes=_peak(host_samples, "mlx_active_bytes"),
                mlx_cache_peak_bytes=_peak(host_samples, "mlx_cache_bytes"),
                system_used_peak_bytes=_peak(host_samples, "system_used_bytes"),
                error=_error_text(request.get("error") or result.get("modelError")),
            )
        )
    return observations


def write_useful_task_bundle(
    output_directory: Path,
    observations: list[UsefulTaskObservation],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    dictionaries = [observation.to_dict() for observation in observations]
    (output_directory / "useful-tasks.json").write_text(
        json.dumps(dictionaries, indent=2) + "\n"
    )
    with (output_directory / "useful-tasks.csv").open("w", newline="") as handle:
        field_names = [field.name for field in fields(UsefulTaskObservation)]
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(dictionaries)
    completed = [item for item in observations if item.request_status == "completed"]
    scored = [item for item in observations if item.score is not None]
    summary = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "observation_count": len(observations),
        "completed_request_count": len(completed),
        "failed_request_count": len(observations) - len(completed),
        "scored_task_count": len(scored),
        "mean_score": (
            sum(item.score for item in scored if item.score is not None) / len(scored)
            if scored
            else None
        ),
        "max_prompt_tokens": max(
            (item.prompt_tokens or 0 for item in observations), default=0
        ),
        "source_run_ids": sorted({item.source_run_id for item in observations}),
        "models": sorted({item.model for item in observations}),
        "benchmarks": sorted({item.benchmark for item in observations}),
    }
    (output_directory / "useful-task-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    _write_useful_task_report(output_directory, observations, summary)
    _write_useful_task_dashboard(output_directory, observations, summary)
    return summary


def _write_useful_task_report(
    output_directory: Path,
    observations: list[UsefulTaskObservation],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Useful-task context observations",
        "",
        f"- Observations: **{summary['observation_count']:,}**",
        f"- Completed requests: **{summary['completed_request_count']:,}**",
        f"- Maximum prompt: **{summary['max_prompt_tokens']:,} tokens**",
        f"- Mean task score: **{summary['mean_score'] if summary['mean_score'] is not None else 'not available'}**",
        "",
        "[Open the useful-task dashboard](useful-task-dashboard.html)",
        "",
        "| Task | Prompt tokens | Generation tok/s | Duration | Score | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for observation in observations:
        generation_rate = observation.generation_tokens_per_second
        duration = observation.total_time_seconds or observation.request_duration_seconds
        lines.append(
            f"| {observation.attempt_id} | {observation.prompt_tokens or 'n/a'} | "
            f"{generation_rate if generation_rate is not None else 'n/a'} | "
            f"{duration if duration is not None else 'n/a'} | "
            f"{observation.score if observation.score is not None else 'n/a'} | "
            f"{observation.request_status} |"
        )
    lines += [
        "",
        "These independent useful-task observations are not persistent-cache capacity phases. "
        "Use the capacity-tail workflow for synthetic continuously growing context.",
        "",
    ]
    (output_directory / "report.md").write_text("\n".join(lines))


def _write_useful_task_dashboard(
    output_directory: Path,
    observations: list[UsefulTaskObservation],
    summary: dict[str, Any],
) -> None:
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item.attempt_id)}</td>"
        f"<td>{item.prompt_tokens if item.prompt_tokens is not None else 'n/a'}</td>"
        f"<td>{item.generation_tokens_per_second if item.generation_tokens_per_second is not None else 'n/a'}</td>"
        f"<td>{item.total_time_seconds if item.total_time_seconds is not None else item.request_duration_seconds or 'n/a'}</td>"
        f"<td>{item.score if item.score is not None else 'n/a'}</td>"
        f"<td>{html.escape(item.request_status)}</td>"
        "</tr>"
        for item in observations
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Useful-task context observations</title><style>
body{{margin:0;background:#f4f0e8;color:#29261f;font:15px/1.5 "Avenir Next",sans-serif}}main{{max-width:1120px;margin:auto;padding:40px 20px}}h1{{font:600 42px/1.05 Georgia,serif;letter-spacing:0}}.stats{{display:flex;gap:32px;border-block:2px solid #29261f;padding:18px 0;margin:28px 0 40px}}.stats b{{display:block;font-size:22px}}table{{border-collapse:collapse;width:100%;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid #d8d0c2;text-align:right}}th:first-child,td:first-child,th:last-child,td:last-child{{text-align:left}}@media(max-width:700px){{main{{padding:24px 10px}}.stats{{display:grid;grid-template-columns:1fr 1fr}}table{{font-size:12px}}th,td{{padding:8px 5px}}}}
</style></head><body><main><h1>Useful-task context observations</h1><div class="stats"><span>Observations<b>{summary['observation_count']:,}</b></span><span>Completed<b>{summary['completed_request_count']:,}</b></span><span>Max prompt<b>{summary['max_prompt_tokens']:,} tokens</b></span></div><table><thead><tr><th>Task attempt</th><th>Prompt tokens</th><th>Generation tok/s</th><th>Duration (s)</th><th>Score</th><th>Status</th></tr></thead><tbody>{rows}</tbody></table></main></body></html>"""
    (output_directory / "useful-task-dashboard.html").write_text(document)