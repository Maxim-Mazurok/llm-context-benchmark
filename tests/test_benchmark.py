from __future__ import annotations

import csv
import json
import struct
import sys
from dataclasses import asdict
from types import ModuleType

import pytest

from llm_context_benchmark.adapters.mlx_lm import MLXLMAdapter
from llm_context_benchmark.adapters.mock import MockAdapter
from llm_context_benchmark.cli import (
    _apply_resume_settings,
    _strip_batch_arguments,
    build_parser,
    main,
)
from llm_context_benchmark.dashboard_server import move_run_to_trash
from llm_context_benchmark.model_discovery import (
    discover_omlx_models,
    safe_model_slug,
)
from llm_context_benchmark.runner import BenchmarkConfig, BenchmarkRunner
from llm_context_benchmark.speculative import (
    choose_speculative_plan,
    configured_num_draft_tokens,
    find_mtp_sidecar,
    has_embedded_mtp,
    plans_for_models,
)
from llm_context_benchmark.useful_tasks import (
    load_workbench_observations,
    write_useful_task_bundle,
)


def test_mtp_generator_disables_wired_memory(monkeypatch):
    wired_limits = []

    class FakeBatchGenerator:
        def __init__(self, model, **arguments):
            self.model = model
            self.arguments = arguments

    mlx_module = ModuleType("mlx")
    mlx_core_module = ModuleType("mlx.core")
    mlx_core_module.set_wired_limit = wired_limits.append
    mlx_module.core = mlx_core_module
    mlx_lm_module = ModuleType("mlx_lm")
    mlx_lm_generate_module = ModuleType("mlx_lm.generate")
    mlx_lm_generate_module.BatchGenerator = FakeBatchGenerator
    mlx_lm_module.generate = mlx_lm_generate_module
    monkeypatch.setitem(sys.modules, "mlx", mlx_module)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core_module)
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm_module)
    monkeypatch.setitem(sys.modules, "mlx_lm.generate", mlx_lm_generate_module)

    adapter = MLXLMAdapter("model", speculative_backend="omlx-mtp")
    adapter.model = object()

    generator = adapter._create_mtp_generator()

    assert generator.model is adapter.model
    assert generator.arguments == {
        "prefill_step_size": 2048,
        "completion_batch_size": 1,
        "prefill_batch_size": 1,
    }
    assert wired_limits == [0]


def test_workbench_telemetry_import_preserves_task_context_observations(tmp_path):
    source = tmp_path / "workbench-run"
    source.mkdir()
    (source / "run.json").write_text(
        json.dumps({"id": "run-1", "benchmark": "humaneval", "model": "model-a"})
    )
    (source / "results.json").write_text(
        json.dumps(
            [
                {
                    "taskId": "HumanEval/0",
                    "attemptId": "HumanEval/0::pass-1",
                    "passNumber": 1,
                    "passed": True,
                    "score": 1,
                    "activeDurationMilliseconds": 2500,
                }
            ]
        )
    )
    (source / "telemetry.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "runId": "run-1",
                "benchmark": "humaneval",
                "model": "model-a",
                "sessions": [
                    {
                        "session_id": "segment-1",
                        "requests": [
                            {
                                "request_id": "HumanEval/0::pass-1",
                                "status": "completed",
                                "duration_seconds": 2.25,
                                "usage": {
                                    "prompt_tokens": 1200,
                                    "completion_tokens": 40,
                                    "prompt_tokens_details": {"cached_tokens": 100},
                                    "total_time": 2.2,
                                    "time_to_first_token": 0.4,
                                    "prompt_eval_duration": 0.3,
                                    "generation_duration": 1.8,
                                    "prompt_tokens_per_second": 4000,
                                    "generation_tokens_per_second": 22.2,
                                },
                                "host_samples": [
                                    {
                                        "physical_footprint_bytes": 100,
                                        "mlx_active_bytes": 200,
                                        "mlx_cache_bytes": 50,
                                        "system_used_bytes": 500,
                                    },
                                    {
                                        "physical_footprint_bytes": 120,
                                        "mlx_active_bytes": 240,
                                        "mlx_cache_bytes": 60,
                                        "system_used_bytes": 550,
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        )
    )

    observations = load_workbench_observations(source)
    output = tmp_path / "imported"
    summary = write_useful_task_bundle(output, observations)

    assert observations[0].attempt_id == "HumanEval/0::pass-1"
    assert observations[0].prompt_tokens == 1200
    assert observations[0].total_tokens == 1240
    assert observations[0].physical_footprint_peak_bytes == 120
    assert observations[0].mlx_active_peak_bytes == 240
    assert summary["observation_count"] == 1
    assert summary["max_prompt_tokens"] == 1200
    assert (output / "useful-tasks.csv").exists()
    assert "Useful-task context observations" in (output / "report.md").read_text()
    assert "HumanEval/0::pass-1" in (output / "useful-task-dashboard.html").read_text()


def test_import_workbench_run_cli_writes_bundle(tmp_path):
    source = tmp_path / "workbench-run"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({"id": "run-1"}))
    (source / "results.json").write_text("[]")
    (source / "telemetry.json").write_text(
        json.dumps({"schemaVersion": 1, "runId": "run-1", "sessions": []})
    )
    output = tmp_path / "observations"

    exit_code = main(
        ["--import-workbench-run", str(source), "--output", str(output)]
    )

    assert exit_code == 0
    assert json.loads((output / "useful-task-summary.json").read_text())[
        "observation_count"
    ] == 0


def test_workbench_import_names_missing_telemetry_artifact(tmp_path):
    (tmp_path / "run.json").write_text("{}")
    (tmp_path / "results.json").write_text("[]")

    with pytest.raises(ValueError, match="missing telemetry.json"):
        load_workbench_observations(tmp_path)


def test_capacity_tail_alias_restores_resume_settings(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_context": 32768}))
    (tmp_path / "run-metadata.json").write_text(
        json.dumps({"model": "mock", "adapter": "mock"})
    )
    (tmp_path / "checkpoint.json").write_text("{}")
    parser = build_parser()
    arguments = parser.parse_args(["--capacity-tail", str(tmp_path)])
    arguments.resume = arguments.capacity_tail

    _apply_resume_settings(
        parser, arguments, ["--capacity-tail", str(tmp_path)]
    )

    assert arguments.resume == tmp_path
    assert arguments.max_context == 32768


def test_draft_depth_uses_omlx_model_setting_then_falls_back(tmp_path):
    model = tmp_path / "Qwen3.6-27B-MLX"
    model.mkdir()
    settings = tmp_path / "model_settings.json"
    settings.write_text(
        json.dumps(
            {
                "models": {
                    "qwen3.6-27b-mlx": {"mtp_num_draft_tokens": 5},
                }
            }
        )
    )

    assert configured_num_draft_tokens(model, settings) == 5
    settings.write_text(json.dumps({"models": {model.name: {}}}))
    assert configured_num_draft_tokens(model, settings) == 3


def test_default_policy_does_not_stop_for_speed_or_pressure():
    config = BenchmarkConfig()
    assert config.severe_decode_ratio is None
    assert config.stop_on_critical_pressure is False


def test_resume_can_clear_saved_context_cap(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"max_context": 32768}))
    (tmp_path / "run-metadata.json").write_text(
        json.dumps({"model": "mock", "adapter": "mock"})
    )
    (tmp_path / "checkpoint.json").write_text("{}")
    argv = ["--resume", str(tmp_path), "--no-max-context", "--adapter", "mock"]
    parser = build_parser()
    args = parser.parse_args(argv)

    _apply_resume_settings(parser, args, argv)

    assert args.max_context is None


def test_mock_benchmark_runs_end_to_end(tmp_path):
    config = BenchmarkConfig(
        chunk_tokens=50,
        short_decode_tokens=10,
        long_decode_tokens=20,
        long_decode_interval=150,
        sample_interval_ms=10,
        max_context=210,
        stop_on_critical_pressure=False,
    )
    summary = BenchmarkRunner(MockAdapter(context_limit=210), config, tmp_path).run()

    assert summary["stop_reason"] == "runtime_context_limit"
    assert summary["max_tested_context_tokens"] == 210
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "memory-vs-context.svg").exists()
    assert (tmp_path / "throughput-vs-context.svg").exists()
    assert (tmp_path / "prefill-speed-vs-context.svg").exists()
    assert (tmp_path / "decode-speed-vs-context.svg").exists()
    assert (tmp_path / "memory-indicators-vs-context.svg").exists()
    dashboard = (tmp_path / "dashboard.html").read_text()
    assert "Prefill speed" in dashboard
    assert "Decode speed" in dashboard
    assert "Memory indicators" in dashboard
    assert 'id="deleteRun"' in dashboard
    assert 'id="deleteDialog"' in dashboard
    assert "Move this run to Trash?" in dashboard

    phases = json.loads((tmp_path / "phases.json").read_text())
    assert {p["decode_kind"] for p in phases if p["phase"] == "decode"} == {
        "short",
        "long",
    }
    decodes = [p for p in phases if p["phase"] == "decode"]
    assert decodes[0]["decode_kind"] == "long"
    assert decodes[-1]["decode_kind"] == "long"
    assert decodes[0]["tokens"] == config.long_decode_tokens
    assert decodes[-1]["tokens"] == config.long_decode_tokens
    assert summary["baseline_decode_kind"] == "long"
    assert summary["baseline_decode_context_tokens"] == decodes[0]["context_end_tokens"]
    prefills = [p for p in phases if p["phase"] == "prefill"]
    assert prefills[0]["cache_catchup_tokens"] == 0
    assert all(p["cache_catchup_tokens"] == 1 for p in prefills[1:])

    with (tmp_path / "samples.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert {row["phase"] for row in rows} >= {"prefill", "decode"}


def test_raw_jsonl_is_valid_and_incremental(tmp_path):
    config = BenchmarkConfig(
        chunk_tokens=20,
        short_decode_tokens=4,
        long_decode_tokens=6,
        long_decode_interval=30,
        sample_interval_ms=10,
        max_context=50,
        stop_on_critical_pressure=False,
    )
    BenchmarkRunner(MockAdapter(context_limit=50), config, tmp_path).run()
    lines = (tmp_path / "samples.jsonl").read_text().splitlines()
    assert len(lines) >= 4
    assert all("context_tokens" in json.loads(line) for line in lines)


def test_completed_run_resumes_from_checkpoint_and_appends_samples(tmp_path):
    first = BenchmarkConfig(
        chunk_tokens=50,
        short_decode_tokens=5,
        long_decode_tokens=10,
        long_decode_interval=100,
        sample_interval_ms=10,
        max_context=120,
    )
    BenchmarkRunner(MockAdapter(context_limit=300), first, tmp_path).run()
    first_phase_count = len(json.loads((tmp_path / "phases.json").read_text()))

    second = BenchmarkConfig(**{**asdict(first), "max_context": 220})
    summary = BenchmarkRunner(
        MockAdapter(context_limit=300), second, tmp_path, resume=True
    ).run()

    assert summary["max_tested_context_tokens"] == 220
    assert len(json.loads((tmp_path / "phases.json").read_text())) > first_phase_count
    assert json.loads((tmp_path / "run-state.json").read_text())["resumable"]
    header = (tmp_path / "samples.csv").read_text().splitlines()[0]
    assert (tmp_path / "samples.csv").read_text().count(header) == 1


def test_interrupt_rolls_checkpoint_back_to_last_completed_cycle(tmp_path):
    class InterruptAfterOneCycle(MockAdapter):
        cycles = 0

        def append_and_decode(self, input_tokens, max_tokens, on_prefill_complete, on_token):
            self.cycles += 1
            if self.cycles == 1:
                return super().append_and_decode(
                    input_tokens, max_tokens, on_prefill_complete, on_token
                )
            on_prefill_complete()
            on_token(999, 0, 1.0)
            raise KeyboardInterrupt

    config = BenchmarkConfig(
        chunk_tokens=50,
        short_decode_tokens=5,
        long_decode_tokens=10,
        long_decode_interval=100,
        sample_interval_ms=10,
        max_context=200,
    )
    summary = BenchmarkRunner(
        InterruptAfterOneCycle(context_limit=200), config, tmp_path
    ).run()
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())

    assert summary["stop_reason"] == "user_interrupt"
    assert checkpoint["context_tokens"] == len(checkpoint["tokens"])
    assert checkpoint["context_tokens"] == summary["max_tested_context_tokens"]
    assert json.loads((tmp_path / "run-state.json").read_text())["resumable"]


def test_delete_moves_only_valid_run_to_trash(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "test-run"
    trash_dir = tmp_path / "trash"
    run_dir.mkdir(parents=True)
    (run_dir / "summary.json").write_text("{}")
    (run_dir / "config.json").write_text("{}")
    (run_dir / "dashboard.html").write_text("dashboard")

    destination = move_run_to_trash(run_dir, trash_dir=trash_dir, runs_root=runs_root)

    assert not run_dir.exists()
    assert destination.parent == trash_dir.resolve()
    assert (destination / "summary.json").exists()


def _write_model(
    root, relative, *, model_type="llama", architecture="LlamaForCausalLM"
):
    directory = root / relative
    directory.mkdir(parents=True)
    (directory / "config.json").write_text(
        json.dumps({"model_type": model_type, "architectures": [architecture]})
    )
    (directory / "model.safetensors").write_bytes(b"weights")
    return directory


def test_omlx_discovery_excludes_auxiliary_models_but_keeps_main_mtp_sidecar(
    tmp_path,
):
    models = tmp_path / "models"
    main = _write_model(models, "org/Main-4bit")
    (main / "mtp.safetensors").write_bytes(b"draft weights")
    _write_model(models, "org/Main-MTP-4bit", model_type="qwen_mtp")
    _write_model(models, "org/Main-assistant-bf16")
    _write_model(
        models,
        "org/Embedding",
        model_type="xlm-roberta",
        architecture="XLMRobertaModel",
    )

    included, excluded = discover_omlx_models(models)

    assert [model.relative_name for model in included] == ["org/Main-4bit"]
    assert {model.relative_name for model in excluded} == {
        "org/Embedding",
        "org/Main-MTP-4bit",
        "org/Main-assistant-bf16",
    }


def test_safe_model_slug_preserves_org_and_model_identity():
    assert safe_model_slug("mlx-community/Model 4-bit") == (
        "mlx-community--Model--4-bit"
    )


def test_omlx_discovery_follows_provider_directory_symlinks(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    shared_provider = tmp_path / "shared-provider"
    _write_model(shared_provider, "Shared-Model")
    (models / "shared").symlink_to(shared_provider, target_is_directory=True)

    included, _excluded = discover_omlx_models(models)

    assert [model.relative_name for model in included] == ["shared/Shared-Model"]


def test_batch_child_arguments_remove_parent_only_options():
    assert _strip_batch_arguments(
        [
            "--all-omlx-models",
            "--omlx-models-dir=/models",
            "--output",
            "/runs",
            "--max-context",
            "65536",
        ]
    ) == ["--max-context", "65536"]


def test_speculative_discovery_prefers_embedded_mtp(tmp_path):
    models = tmp_path / "models"
    target = _write_model(
        models,
        "org/Target",
        model_type="qwen3_5",
        architecture="QwenForCausalLM",
    )
    config = json.loads((target / "config.json").read_text())
    config["mtp_num_hidden_layers"] = 1
    (target / "config.json").write_text(json.dumps(config))
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.fc.weight": "model.safetensors"}})
    )
    (target / "tokenizer.json").write_text('{"same": true}')

    included, _excluded = discover_omlx_models(models)
    plan = choose_speculative_plan(
        target,
        included,
        tmp_path / "missing-settings.json",
    )

    assert has_embedded_mtp(target)
    assert plan is not None
    assert plan.backend == "omlx-mtp"


def test_speculative_discovery_chooses_smallest_identical_tokenizer(tmp_path):
    models = tmp_path / "models"
    target = _write_model(models, "org/Target")
    draft = _write_model(models, "org/Small")
    for path in (target, draft):
        (path / "tokenizer.json").write_text('{"same": true}')
    (target / "model.safetensors").write_bytes(b"large target")
    (draft / "model.safetensors").write_bytes(b"d")

    included, _excluded = discover_omlx_models(models)
    plans = plans_for_models(included, tmp_path / "missing-settings.json")

    assert plans[target][0].backend == "mlx-draft"
    assert plans[target][0].draft_model == draft


def test_speculative_discovery_accepts_prefixed_mtp_sidecar(tmp_path):
    models = tmp_path / "models"
    target = _write_model(
        models,
        "org/Target",
        model_type="qwen3_5",
        architecture="QwenForCausalLM",
    )
    config = json.loads((target / "config.json").read_text())
    config["mtp_num_hidden_layers"] = 1
    (target / "config.json").write_text(json.dumps(config))
    header = json.dumps(
        {
            "__metadata__": {"format": "mlx"},
            "mtp.fc.weight": {
                "dtype": "F32",
                "shape": [0],
                "data_offsets": [0, 0],
            },
        }
    ).encode()
    sidecar = target / "mtp.safetensors"
    sidecar.write_bytes(struct.pack("<Q", len(header)) + header)

    included, _excluded = discover_omlx_models(models)
    plan = choose_speculative_plan(
        target,
        included,
        tmp_path / "missing-settings.json",
    )

    assert find_mtp_sidecar(target) == sidecar
    assert plan is not None
    assert plan.backend == "omlx-mtp"
    assert plan.mtp_sidecar == sidecar


def test_speculative_discovery_reuses_compatible_mtp_helper_for_targets(tmp_path):
    models = tmp_path / "models"
    targets = [
        _write_model(
            models,
            relative,
            model_type="qwen3_5",
            architecture="QwenForCausalLM",
        )
        for relative in ("original/Qwen3.8-27B", "custom/Qwen3.8-27B-Uncensored")
    ]
    helper = _write_model(
        models,
        "mlx-community/Qwen3.8-27B-MTP-4bit",
        model_type="qwen3_5_mtp",
        architecture="QwenMTPModel",
    )
    dimensions = {
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 64,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "vocab_size": 248320,
        "mtp_num_hidden_layers": 1,
    }
    for path in [*targets, helper]:
        config = json.loads((path / "config.json").read_text())
        config.update(dimensions)
        (path / "config.json").write_text(json.dumps(config))
        (path / "tokenizer.json").write_text('{"same": true}')

    included, excluded = discover_omlx_models(models)
    plans = plans_for_models(
        included, tmp_path / "missing-settings.json", excluded
    )

    assert helper not in [model.path for model in included]
    for target in targets:
        assert plans[target][0].backend == "omlx-mtp"
        assert plans[target][0].mtp_helper == helper


def test_speculative_discovery_rejects_mismatched_mtp_helper(tmp_path):
    models = tmp_path / "models"
    target = _write_model(models, "org/Target", model_type="qwen3_5")
    helper = _write_model(models, "org/Other-MTP", model_type="qwen3_5_mtp")
    for path, hidden_size in ((target, 5120), (helper, 4096)):
        config = json.loads((path / "config.json").read_text())
        config.update({"hidden_size": hidden_size, "mtp_num_hidden_layers": 1})
        (path / "config.json").write_text(json.dumps(config))

    included, excluded = discover_omlx_models(models)
    plans = plans_for_models(
        included, tmp_path / "missing-settings.json", excluded
    )

    assert not plans[target]


def test_speculative_discovery_matches_moe_mtp_helper_from_symlinked_provider(
    tmp_path,
):
    models = tmp_path / "models"
    models.mkdir()
    lmstudio_provider = tmp_path / "lmstudio-provider"
    target_real = _write_model(
        lmstudio_provider,
        "Qwen3.6-35B-A3B-4bit",
        model_type="qwen3_5_moe",
    )
    helper_real = _write_model(
        lmstudio_provider,
        "Qwen3.6-35B-A3B-MTP-4bit",
        model_type="qwen3_5_mtp",
    )
    common = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "mtp_num_hidden_layers": 1,
    }
    for path in (target_real, helper_real):
        config = json.loads((path / "config.json").read_text())
        config["text_config"] = common
        (path / "config.json").write_text(json.dumps(config))
    (models / "lmstudio-community").symlink_to(
        lmstudio_provider, target_is_directory=True
    )

    included, excluded = discover_omlx_models(models)
    plans = plans_for_models(
        included, tmp_path / "missing-settings.json", excluded
    )
    target = next(
        model
        for model in included
        if model.relative_name.endswith("Qwen3.6-35B-A3B-4bit")
    )

    assert target.path.resolve() == target_real.resolve()
    assert plans[target.path][0].mtp_helper is not None
    assert plans[target.path][0].mtp_helper.resolve() == helper_real.resolve()
    assert (
        choose_speculative_plan(
            target.path,
            included,
            tmp_path / "missing-settings.json",
            helpers=excluded,
        ).mtp_helper.resolve()
        == helper_real.resolve()
    )


def test_external_draft_rejects_nontrimmable_linear_attention(tmp_path):
    models = tmp_path / "models"
    target = _write_model(models, "org/Target")
    draft = _write_model(models, "org/Draft")
    for path in (target, draft):
        config = json.loads((path / "config.json").read_text())
        config["layer_types"] = ["linear_attention", "full_attention"]
        (path / "config.json").write_text(json.dumps(config))
        (path / "tokenizer.json").write_text('{"same": true}')
    (target / "model.safetensors").write_bytes(b"large target")

    included, _excluded = discover_omlx_models(models)

    assert (
        choose_speculative_plan(
            target,
            included,
            tmp_path / "missing-settings.json",
            backend="mlx-draft",
            draft_model=draft,
        )
        is None
    )
