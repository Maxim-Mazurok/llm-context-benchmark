from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .adapters import MLXLMAdapter, MockAdapter, UnslothLlamaCppAdapter
from .model_discovery import LocalModel, discover_omlx_models, safe_model_slug
from .runner import BenchmarkConfig, BenchmarkRunner
from .speculative import (
    SpeculativePlan,
    choose_speculative_plan,
    configured_num_draft_tokens,
    plans_for_models,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-context-bench",
        description="Measure maximum usable context with one continuously growing KV cache.",
    )
    parser.add_argument(
        "--model", help="Hugging Face model id or local MLX model directory"
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume a stopped run from its last completed-cycle checkpoint",
    )
    parser.add_argument(
        "--capacity-tail",
        type=Path,
        help=(
            "Resume a synthetic persistent-cache run as the capacity tail; "
            "use --max-context to stage its next endpoint"
        ),
    )
    parser.add_argument(
        "--import-workbench-run",
        type=Path,
        help="Import run.json, results.json, and telemetry.json from an Eval Workbench run",
    )
    parser.add_argument(
        "--recover-run",
        type=Path,
        help="Recover charts and summaries from raw samples of an older interrupted run",
    )
    parser.add_argument(
        "--upgrade-run-memory",
        type=Path,
        help="Recompute peak memory metrics for a completed legacy run",
    )
    batch_group = parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        "--all-omlx-models",
        action="store_true",
        help="Benchmark every eligible model installed in the OMLX model library",
    )
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    batch_group.add_argument(
        "--list-omlx-models",
        action="store_true",
        help="List models batch mode would include or exclude, then exit",
    )
    parser.add_argument(
        "--omlx-models-dir",
        type=Path,
        default=Path.home() / ".omlx" / "models",
        help="OMLX model library (default: ~/.omlx/models)",
    )
    parser.add_argument(
        "--omlx-model-settings",
        type=Path,
        default=Path.home() / ".omlx" / "model_settings.json",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Enable speculative decoding (auto-select backend/drafter by default)",
    )
    parser.add_argument(
        "--also-speculative",
        action="store_true",
        help="In OMLX batch mode, benchmark raw and speculative decoding",
    )
    parser.add_argument(
        "--speculative-backend",
        choices=("auto", "mlx-draft", "mlx-vlm-mtp", "omlx-mtp"),
        default="auto",
        help="Speculative backend selection (default: auto)",
    )
    parser.add_argument(
        "--draft-model",
        type=Path,
        help="Explicit tokenizer-compatible MLX draft model directory",
    )
    parser.add_argument("--mtp-sidecar", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--mtp-helper", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Maximum speculative draft depth (default: OMLX setting, otherwise 3)",
    )
    parser.add_argument(
        "--adapter",
        choices=("mlx-lm", "mock", "unsloth-llama.cpp"),
        default="mlx-lm",
    )
    parser.add_argument(
        "--unsloth-server",
        help="Path to Unsloth's llama-server executable",
    )
    parser.add_argument(
        "--serve-run",
        type=Path,
        help="Serve an existing run dashboard locally with run deletion enabled",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the browser when used with --serve-run",
    )
    parser.add_argument(
        "--output", type=Path, help="Output directory (default: runs/<timestamp>)"
    )
    parser.add_argument("--chunk-tokens", type=int, default=5_000)
    parser.add_argument("--short-decode-tokens", type=int, default=256)
    parser.add_argument("--long-decode-tokens", type=int, default=1_000)
    parser.add_argument("--long-decode-interval", type=int, default=25_000)
    parser.add_argument("--sample-ms", type=int, default=100)
    context_group = parser.add_mutually_exclusive_group()
    context_group.add_argument(
        "--max-context",
        type=int,
        help="Optional cap; otherwise use model limit when known",
    )
    context_group.add_argument(
        "--no-max-context",
        action="store_true",
        help="On resume, clear the previous cap and continue to the runtime limit",
    )
    parser.add_argument("--swap-stop-gib", type=float, default=4.0)
    parser.add_argument(
        "--severe-decode-ratio",
        type=float,
        help="Optional speed-based stop ratio; disabled by default",
    )
    parser.add_argument("--severe-decode-consecutive", type=int, default=2)
    parser.add_argument("--practical-decode-ratio", type=float, default=0.8)
    parser.add_argument("--practical-swap-growth-gib", type=float, default=1.0)
    parser.add_argument("--no-swap-epsilon-mib", type=float, default=64.0)
    pressure_group = parser.add_mutually_exclusive_group()
    pressure_group.add_argument(
        "--stop-on-critical-pressure",
        action="store_true",
        help="Stop on critical macOS memory pressure (disabled by default)",
    )
    pressure_group.add_argument(
        "--continue-on-critical-pressure",
        dest="stop_on_critical_pressure",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(stop_on_critical_pressure=False)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument("--kv-bits", type=int, choices=(4, 8))
    parser.add_argument("--kv-group-size", type=int, default=64)
    parser.add_argument("--seed-text-file", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def _positive(parser: argparse.ArgumentParser, name: str, value: int) -> None:
    if value <= 0:
        parser.error(f"{name} must be greater than zero")


def _option_present(argv: list[str], option: str) -> bool:
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


def _apply_resume_settings(
    parser: argparse.ArgumentParser, args: argparse.Namespace, raw_argv: list[str]
) -> None:
    if args.resume is None:
        return
    run_dir = args.resume.expanduser().resolve()
    try:
        config = json.loads((run_dir / "config.json").read_text())
        metadata = json.loads((run_dir / "run-metadata.json").read_text())
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"Cannot resume {run_dir}: {exc}")
    if not (run_dir / "checkpoint.json").is_file():
        parser.error(
            "This run has no checkpoint. It can be viewed after recovery but "
            "cannot be resumed exactly."
        )
    if not _option_present(raw_argv, "--model"):
        args.model = metadata.get("model")
    if not _option_present(raw_argv, "--adapter") and metadata.get("adapter"):
        args.adapter = metadata["adapter"]
    config_options = {
        "chunk_tokens": "--chunk-tokens",
        "short_decode_tokens": "--short-decode-tokens",
        "long_decode_tokens": "--long-decode-tokens",
        "long_decode_interval": "--long-decode-interval",
        "sample_interval_ms": "--sample-ms",
        "max_context": "--max-context",
        "swap_stop_gib": "--swap-stop-gib",
        "severe_decode_ratio": "--severe-decode-ratio",
        "severe_decode_consecutive": "--severe-decode-consecutive",
        "practical_decode_ratio": "--practical-decode-ratio",
        "practical_swap_growth_gib": "--practical-swap-growth-gib",
        "no_swap_epsilon_mib": "--no-swap-epsilon-mib",
        "stop_on_critical_pressure": "--stop-on-critical-pressure",
    }
    for field, option in config_options.items():
        if (
            field in config
            and not _option_present(raw_argv, option)
            and not (field == "max_context" and args.no_max_context)
        ):
            setattr(args, field, config[field])
    if metadata.get("decode_mode") == "speculative":
        args.speculative = True
        if not _option_present(raw_argv, "--speculative-backend"):
            args.speculative_backend = metadata.get("speculative_backend") or "auto"
        if args.draft_model is None and metadata.get("draft_model"):
            args.draft_model = Path(metadata["draft_model"])
        if args.mtp_sidecar is None and metadata.get("mtp_sidecar"):
            args.mtp_sidecar = Path(metadata["mtp_sidecar"])
        if args.mtp_helper is None and metadata.get("mtp_helper"):
            args.mtp_helper = Path(metadata["mtp_helper"])
        if args.num_draft_tokens is None and metadata.get("num_draft_tokens"):
            args.num_draft_tokens = int(metadata["num_draft_tokens"])
    args.output = run_dir


def _print_discovered_models(
    included: list[LocalModel],
    excluded: list[LocalModel],
    *,
    root: Path,
    speculative_plans: dict[Path, list[SpeculativePlan]],
) -> None:
    print(f"OMLX model library: {root.expanduser().resolve()}")
    print(f"Eligible models ({len(included)}):")
    for model in included:
        plans = speculative_plans.get(model.path, [])
        capabilities = "; ".join(plan.description for plan in plans)
        suffix = f"; speculative: {capabilities}" if capabilities else ""
        print(f"  + {model.relative_name} [raw{suffix}]")
    print(f"Excluded models ({len(excluded)}):")
    for model in excluded:
        print(f"  - {model.relative_name} ({model.exclusion_reason})")


def _strip_batch_arguments(argv: list[str]) -> list[str]:
    value_options = {
        "--model",
        "--output",
        "--omlx-models-dir",
        "--omlx-model-settings",
        "--speculative-backend",
        "--draft-model",
        "--mtp-sidecar",
        "--mtp-helper",
    }
    flag_options = {
        "--all-omlx-models",
        "--list-omlx-models",
        "--speculative",
        "--also-speculative",
        "--json",
    }
    result: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value in flag_options:
            index += 1
            continue
        if value in value_options:
            index += 2
            continue
        if any(value.startswith(f"{option}=") for option in value_options):
            index += 1
            continue
        result.append(value)
        index += 1
    return result


def _unique_batch_output(root: Path, relative_name: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    base = root / f"{stamp}--{safe_model_slug(relative_name)}"
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = root / f"{base.name}-{suffix}"
        suffix += 1
    return candidate


def _run_omlx_batch(args: argparse.Namespace, raw_argv: list[str]) -> int:
    included, excluded = discover_omlx_models(args.omlx_models_dir)
    speculative_plans = plans_for_models(
        included, args.omlx_model_settings, excluded
    )
    if args.json:
        print(
            json.dumps(
                {
                    "root": str(args.omlx_models_dir.expanduser().resolve()),
                    "models": [
                        {
                            "path": str(model.path),
                            "relativeName": model.relative_name,
                            "speculative": [
                                {
                                    "backend": plan.backend,
                                    "description": plan.description,
                                }
                                for plan in speculative_plans.get(model.path, [])
                            ],
                        }
                        for model in included
                    ],
                    "excluded": [
                        {
                            "path": str(model.path),
                            "relativeName": model.relative_name,
                            "reason": model.exclusion_reason,
                        }
                        for model in excluded
                    ],
                },
                separators=(",", ":"),
            )
        )
    else:
        _print_discovered_models(
            included,
            excluded,
            root=args.omlx_models_dir,
            speculative_plans=speculative_plans,
        )
    if args.list_omlx_models:
        return 0
    if not included:
        print("No eligible local OMLX models were found.", file=sys.stderr)
        return 1

    selected: list[tuple[LocalModel, SpeculativePlan | None]] = []
    for model in included:
        if not args.speculative or args.also_speculative:
            selected.append((model, None))
        if args.speculative or args.also_speculative:
            plan = choose_speculative_plan(
                model.path,
                included,
                args.omlx_model_settings,
                backend=args.speculative_backend,
                draft_model=args.draft_model,
                helpers=excluded,
            )
            if plan is None:
                print(
                    f"No speculative variant for {model.relative_name}: no "
                    "compatible backend/drafter was found."
                    + (" The raw variant remains queued." if args.also_speculative else ""),
                    file=sys.stderr,
                )
                continue
            selected.append((model, plan))
    if not selected:
        print("No models match the requested benchmark mode.", file=sys.stderr)
        return 1

    project_root = Path(__file__).resolve().parents[2]
    output_root = (args.output or project_root / "runs").expanduser().resolve()
    child_arguments = _strip_batch_arguments(raw_argv)
    failures: list[str] = []
    for index, (model, plan) in enumerate(selected, start=1):
        mode_name = "spec" if plan else "raw"
        output = _unique_batch_output(output_root, f"{model.relative_name}--{mode_name}")
        print(
            f"\n[{index}/{len(selected)}] Benchmarking {model.relative_name}"
            f"{f' with {plan.description}' if plan else ''}\n"
            f"Output: {output}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "llm_context_benchmark",
            *child_arguments,
            "--model",
            str(model.path),
            "--output",
            str(output),
        ]
        if plan is not None:
            command.extend(
                ["--speculative", "--speculative-backend", plan.backend]
            )
            if plan.draft_model is not None:
                command.extend(["--draft-model", str(plan.draft_model)])
            if plan.mtp_sidecar is not None:
                command.extend(["--mtp-sidecar", str(plan.mtp_sidecar)])
            if plan.mtp_helper is not None:
                command.extend(["--mtp-helper", str(plan.mtp_helper)])
            if not _option_present(raw_argv, "--num-draft-tokens"):
                command.extend(
                    [
                        "--num-draft-tokens",
                        str(
                            configured_num_draft_tokens(
                                model.path, args.omlx_model_settings
                            )
                        ),
                    ]
                )
        process = subprocess.Popen(command)
        try:
            return_code = process.wait()
        except KeyboardInterrupt:
            print("\nStopping the active benchmark and saving its checkpoint…")
            try:
                return_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.terminate()
                return_code = process.wait(timeout=10)
            print("Batch interrupted; remaining models were not started.")
            return 130
        if return_code != 0:
            failures.append(model.relative_name)
            print(
                f"Continuing after failure in {model.relative_name} "
                f"(exit {return_code}).",
                file=sys.stderr,
            )
    succeeded = len(selected) - len(failures)
    print(f"\nBatch complete: {succeeded} succeeded, {len(failures)} failed.")
    if failures:
        print("Failed models: " + ", ".join(failures), file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.resume and args.capacity_tail:
        parser.error("--resume and --capacity-tail cannot be combined")
    if args.capacity_tail:
        args.resume = args.capacity_tail
    _apply_resume_settings(parser, args, raw_argv)
    if args.import_workbench_run:
        if args.resume or args.model or args.all_omlx_models or args.list_omlx_models:
            parser.error(
                "--import-workbench-run cannot be combined with model or capacity-run options"
            )
        from .useful_tasks import (
            load_workbench_observations,
            write_useful_task_bundle,
        )

        try:
            observations = load_workbench_observations(args.import_workbench_run)
        except ValueError as error:
            parser.error(str(error))
        root = Path(__file__).resolve().parents[2]
        stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
        output = (
            args.output or root / "runs" / f"{stamp}--useful-tasks"
        ).expanduser().resolve()
        summary = write_useful_task_bundle(output, observations)
        print(
            f"Imported {summary['observation_count']:,} useful-task observations "
            f"to {output}"
        )
        print(f"Report: {output / 'report.md'}")
        return 0
    if args.recover_run:
        from .recovery import recover_interrupted_run

        try:
            summary = recover_interrupted_run(args.recover_run)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"Recovered {summary['max_tested_context_tokens']:,} tested tokens "
            f"from {args.recover_run.expanduser().resolve()}"
        )
        return 0
    if args.upgrade_run_memory:
        from .recovery import upgrade_legacy_memory_metrics

        try:
            summary = upgrade_legacy_memory_metrics(args.upgrade_run_memory)
        except ValueError as exc:
            parser.error(str(exc))
        print(
            f"Upgraded peak memory metrics through "
            f"{summary['max_tested_context_tokens']:,} tokens in "
            f"{args.upgrade_run_memory.expanduser().resolve()}"
        )
        return 0
    if args.serve_run and (args.all_omlx_models or args.list_omlx_models):
        parser.error("--serve-run cannot be combined with OMLX model discovery options")
    if args.serve_run:
        from .dashboard_server import serve_run_dashboard

        try:
            return serve_run_dashboard(args.serve_run, open_browser=args.open_browser)
        except ValueError as exc:
            parser.error(str(exc))
    if args.model and (args.all_omlx_models or args.list_omlx_models):
        parser.error("--model cannot be combined with OMLX model discovery options")
    if args.resume and (args.all_omlx_models or args.list_omlx_models):
        parser.error("--resume cannot be combined with OMLX model discovery options")
    if args.speculative and args.adapter != "mlx-lm":
        parser.error("--speculative requires --adapter mlx-lm")
    if args.also_speculative and not args.all_omlx_models:
        parser.error("--also-speculative requires --all-omlx-models")
    if (
        args.all_omlx_models or args.list_omlx_models
    ) and args.adapter != "mlx-lm":
        parser.error("OMLX model discovery requires --adapter mlx-lm")
    if args.adapter in ("mlx-lm", "unsloth-llama.cpp") and not (
        args.model or args.all_omlx_models or args.list_omlx_models
    ):
        parser.error(f"--model is required with --adapter {args.adapter}")
    for name in (
        "chunk_tokens",
        "short_decode_tokens",
        "long_decode_tokens",
        "long_decode_interval",
        "sample_ms",
    ):
        _positive(parser, "--" + name.replace("_", "-"), getattr(args, name))
    if args.sample_ms < 10:
        parser.error("--sample-ms must be at least 10 ms")
    if args.max_context is not None and args.max_context <= 0:
        parser.error("--max-context must be greater than zero")
    if args.severe_decode_ratio is not None and not 0 < args.severe_decode_ratio < 1:
        parser.error("--severe-decode-ratio must be between zero and one")
    if args.num_draft_tokens is not None and args.num_draft_tokens <= 0:
        parser.error("--num-draft-tokens must be greater than zero")
    if args.draft_model is not None and not args.speculative:
        parser.error("--draft-model requires --speculative")
    if args.speculative_backend != "auto" and not args.speculative:
        parser.error("--speculative-backend requires --speculative")
    if args.all_omlx_models or args.list_omlx_models:
        return _run_omlx_batch(args, raw_argv)

    speculative_plan = None
    if args.speculative:
        models, excluded = discover_omlx_models(args.omlx_models_dir)
        target_path = Path(args.model).expanduser().resolve()
        speculative_plan = choose_speculative_plan(
            target_path,
            models,
            args.omlx_model_settings,
            backend=args.speculative_backend,
            draft_model=args.draft_model,
            helpers=excluded,
        )
        if speculative_plan is None:
            parser.error(
                "No compatible speculative backend/drafter was found for this "
                "model. Use --list-omlx-models to inspect detected pairings or "
                "provide --draft-model."
            )
        if args.num_draft_tokens is None:
            args.num_draft_tokens = configured_num_draft_tokens(
                target_path, args.omlx_model_settings
            )

    if args.num_draft_tokens is None:
        args.num_draft_tokens = 3

    root = Path(__file__).resolve().parents[2]
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    output = (args.output or root / "runs" / stamp).expanduser().resolve()
    seed_text = args.seed_text_file.read_text() if args.seed_text_file else None
    if args.adapter == "mock":
        adapter = MockAdapter(
            args.model or "mock", context_limit=args.max_context or 100_000
        )
    elif args.adapter == "unsloth-llama.cpp":
        adapter = UnslothLlamaCppAdapter(
            args.model,
            context_size=args.max_context,
            server_path=args.unsloth_server,
            seed_text=seed_text,
        )
    else:
        adapter = MLXLMAdapter(
            args.model,
            speculative_backend=(
                speculative_plan.backend if speculative_plan else None
            ),
            draft_model=(
                str(speculative_plan.draft_model)
                if speculative_plan and speculative_plan.draft_model
                else None
            ),
            mtp_sidecar=(
                str(speculative_plan.mtp_sidecar)
                if speculative_plan and speculative_plan.mtp_sidecar
                else (str(args.mtp_sidecar) if args.mtp_sidecar else None)
            ),
            mtp_helper=(
                str(speculative_plan.mtp_helper)
                if speculative_plan and speculative_plan.mtp_helper
                else (str(args.mtp_helper) if args.mtp_helper else None)
            ),
            num_draft_tokens=args.num_draft_tokens,
            prefill_step_size=args.prefill_step_size,
            kv_bits=args.kv_bits,
            kv_group_size=args.kv_group_size,
            seed_text=seed_text,
            trust_remote_code=args.trust_remote_code,
        )
    config = BenchmarkConfig(
        chunk_tokens=args.chunk_tokens,
        short_decode_tokens=args.short_decode_tokens,
        long_decode_tokens=args.long_decode_tokens,
        long_decode_interval=args.long_decode_interval,
        sample_interval_ms=args.sample_ms,
        max_context=args.max_context,
        swap_stop_gib=args.swap_stop_gib,
        severe_decode_ratio=args.severe_decode_ratio,
        severe_decode_consecutive=args.severe_decode_consecutive,
        practical_decode_ratio=args.practical_decode_ratio,
        practical_swap_growth_gib=args.practical_swap_growth_gib,
        no_swap_epsilon_mib=args.no_swap_epsilon_mib,
        stop_on_critical_pressure=args.stop_on_critical_pressure,
    )
    print(f"Writing benchmark data to {output}", flush=True)
    summary = BenchmarkRunner(adapter, config, output, resume=bool(args.resume)).run()
    print(f"Stopped: {summary['stop_reason']}")
    print(f"Max tested: {summary['max_tested_context_tokens']:,} tokens")
    practical = summary.get("practical_context_tokens")
    print(
        f"Practical context: {practical:,} tokens"
        if practical
        else "Practical context: not established"
    )
    print(f"Report: {output / 'report.md'}")
    return 0 if summary["stop_reason"] not in ("runtime_error",) else 1


if __name__ == "__main__":
    raise SystemExit(main())
