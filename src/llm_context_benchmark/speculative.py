from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

from .model_discovery import LocalModel


@dataclass(frozen=True, slots=True)
class SpeculativePlan:
    backend: str
    source: str
    draft_model: Path | None = None
    mtp_sidecar: Path | None = None
    mtp_helper: Path | None = None

    @property
    def description(self) -> str:
        if self.backend == "omlx-mtp":
            helper = f": {self.mtp_helper.name}" if self.mtp_helper else ""
            return f"OMLX Lightning MTP ({self.source}{helper})"
        if self.backend == "mlx-vlm-mtp":
            draft = self.draft_model.name if self.draft_model else "unknown drafter"
            return f"MLX-VLM MTP: {draft} ({self.source})"
        draft = self.draft_model.name if self.draft_model else "unknown draft"
        return f"MLX draft: {draft} ({self.source})"


def _read_config(model_path: Path) -> dict[str, object]:
    try:
        value = json.loads((model_path / "config.json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _has_mtp_heads(config: dict[str, object]) -> bool:
    configs = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        configs.append(text_config)
    for candidate in configs:
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            try:
                if int(candidate.get(key, 0) or 0) > 0:
                    return True
            except (TypeError, ValueError):
                pass
    mtp_config = config.get("mtp_config")
    if isinstance(mtp_config, dict):
        try:
            return int(mtp_config.get("num_nextn_predict_layers", 0) or 0) > 0
        except (TypeError, ValueError):
            return False
    return False


def _mtp_compatible(config: dict[str, object]) -> bool:
    model_type = str(config.get("model_type") or "")
    return _has_mtp_heads(config) and (
        model_type.startswith(("qwen3_5", "qwen3_6", "deepseek_v4", "nemotron_h"))
        or model_type in {
            "gemma4",
            "gemma4_unified",
            "glm_moe_dsa",
            "inkling",
            "inkling_mm_model",
            "step3p7",
        }
    )


def _config_value(config: dict[str, object], key: str) -> object:
    value = config.get(key)
    if value is not None:
        return value
    text_config = config.get("text_config")
    return text_config.get(key) if isinstance(text_config, dict) else None


def _mtp_architecture_signature(config: dict[str, object]) -> tuple[object, ...]:
    text_config = config.get("text_config")
    text_model_type = (
        str(text_config.get("model_type") or "")
        if isinstance(text_config, dict)
        else ""
    )
    model_type = (
        text_model_type.removesuffix("_text")
        if text_model_type
        else str(config.get("model_type") or "").removesuffix("_mtp")
    )
    return (
        model_type,
        *(
            _config_value(config, key)
            for key in (
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
                "full_attention_interval",
                "linear_conv_kernel_dim",
                "linear_key_head_dim",
                "linear_num_key_heads",
                "linear_num_value_heads",
                "linear_value_head_dim",
                "moe_intermediate_size",
                "num_experts",
                "num_experts_per_tok",
                "shared_expert_intermediate_size",
            )
        ),
    )


def _is_mtp_helper(model: LocalModel) -> bool:
    config = _read_config(model.path)
    model_type = str(config.get("model_type") or "").casefold()
    return model_type.endswith("_mtp") or "mtp" in model.path.name.casefold()


def _find_mtp_helper(
    target_path: Path, helpers: list[LocalModel]
) -> LocalModel | None:
    target_config = _read_config(target_path)
    if not _mtp_compatible(target_config):
        return None
    signature = _mtp_architecture_signature(target_config)
    candidates = [
        helper
        for helper in helpers
        if _is_mtp_helper(helper)
        and _mtp_architecture_signature(_read_config(helper.path)) == signature
    ]
    if not candidates:
        return None
    target_fingerprint = _tokenizer_fingerprint(target_path)
    target_version = _model_version_hint(target_path)
    return min(
        candidates,
        key=lambda helper: (
            _model_version_hint(helper.path) != target_version,
            _tokenizer_fingerprint(helper.path) != target_fingerprint,
            _model_weight_bytes(helper.path),
            helper.relative_name.casefold(),
        ),
    )


def _is_vlm_mtp_pair(target_path: Path, helper_path: Path) -> bool:
    target_config = _read_config(target_path)
    helper_config = _read_config(helper_path)
    target_model_type = str(target_config.get("model_type") or "")
    helper_model_type = str(helper_config.get("model_type") or "")
    if helper_model_type not in {
        "gemma4_assistant",
        "gemma4_unified_assistant",
    } or helper_model_type != f"{target_model_type}_assistant":
        return False
    target_hidden_size = _config_value(target_config, "hidden_size")
    helper_backbone_hidden_size = helper_config.get("backbone_hidden_size")
    if (
        target_hidden_size is None
        or helper_backbone_hidden_size != target_hidden_size
        or _config_value(target_config, "vocab_size")
        != _config_value(helper_config, "vocab_size")
    ):
        return False
    return True


def _find_vlm_mtp_helper(
    target_path: Path, helpers: list[LocalModel]
) -> LocalModel | None:
    candidates = [
        helper
        for helper in helpers
        if _is_vlm_mtp_pair(target_path, helper.path)
    ]
    return (
        min(candidates, key=lambda helper: helper.relative_name.casefold())
        if candidates
        else None
    )


def _model_version_hint(model_path: Path) -> str | None:
    match = re.search(r"qwen\s*([0-9]+(?:\.[0-9]+)?)", model_path.name.casefold())
    return match.group(1) if match else None


def _nextn_prefixes(config: dict[str, object]) -> tuple[str, ...]:
    configs = [config]
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        configs.append(text_config)
    try:
        count = max(int(item.get("num_nextn_predict_layers", 0) or 0) for item in configs)
        main_layers = max(int(item.get("num_hidden_layers", 0) or 0) for item in configs)
    except (TypeError, ValueError):
        return ()
    if count <= 0 or main_layers <= 0:
        return ()
    return tuple(
        prefix
        for index in range(count)
        for prefix in (
            f"model.layers.{main_layers + index}.",
            f"language_model.model.layers.{main_layers + index}.",
            f"model.language_model.layers.{main_layers + index}.",
        )
    )


def _safetensor_keys(path: Path) -> tuple[str, ...]:
    try:
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if header_size > 128 * 1024 * 1024:
                return ()
            header = json.loads(handle.read(header_size))
        return tuple(key for key in header if key != "__metadata__")
    except (OSError, ValueError, struct.error, json.JSONDecodeError):
        return ()


def has_embedded_mtp(model_path: Path) -> bool:
    config = _read_config(model_path)
    if not _mtp_compatible(config):
        return False
    prefixes = (
        "mtp.",
        "language_model.mtp.",
        "model.mtp.",
        "model.language_model.mtp.",
        *_nextn_prefixes(config),
    )
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            keys = tuple((index.get("weight_map") or {}).keys())
        except (OSError, json.JSONDecodeError, AttributeError):
            keys = ()
        return any(key.startswith(prefixes) for key in keys)
    for shard in sorted(model_path.glob("model*.safetensors")):
        if any(key.startswith(prefixes) for key in _safetensor_keys(shard)):
            return True
    return False


def find_mtp_sidecar(model_path: Path) -> Path | None:
    for sidecar in sorted(model_path.glob("*mtp*.safetensors")):
        if any(
            key.startswith(
                (
                    "mtp.",
                    "language_model.mtp.",
                    "model.mtp.",
                    "model.language_model.mtp.",
                )
            )
            for key in _safetensor_keys(sidecar)
        ):
            return sidecar
    return None


def _supports_external_draft(model_path: Path) -> bool:
    config = _read_config(model_path)
    layer_types = config.get("layer_types")
    if not isinstance(layer_types, list):
        text_config = config.get("text_config")
        layer_types = (
            text_config.get("layer_types")
            if isinstance(text_config, dict)
            else None
        )
    if isinstance(layer_types, list) and any(
        str(layer_type) != "full_attention" for layer_type in layer_types
    ):
        return False
    sliding_window = config.get("sliding_window")
    if sliding_window is None and isinstance(config.get("text_config"), dict):
        sliding_window = config["text_config"].get("sliding_window")
    return not bool(sliding_window)


def _tokenizer_fingerprint(model_path: Path) -> str | None:
    tokenizer = model_path / "tokenizer.json"
    if not tokenizer.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with tokenizer.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _model_weight_bytes(model_path: Path) -> int:
    total = 0
    for weight in model_path.glob("model*.safetensors"):
        try:
            total += weight.stat().st_size
        except OSError:
            pass
    return total


def _model_settings(settings_path: Path) -> dict[str, dict[str, object]]:
    try:
        value = json.loads(settings_path.expanduser().read_text()).get("models", {})
    except (OSError, json.JSONDecodeError, AttributeError):
        return {}
    return value if isinstance(value, dict) else {}


def _settings_for_model(
    model_path: Path, settings: dict[str, dict[str, object]]
) -> dict[str, object]:
    basename = model_path.name.casefold()
    for identifier, value in settings.items():
        if str(identifier).casefold() == basename and isinstance(value, dict):
            return value
    return {}


def configured_num_draft_tokens(
    model_path: Path, settings_path: Path, fallback: int = 3
) -> int:
    """Return the per-model OMLX MTP depth, or OMLX's usual default."""
    configured = _settings_for_model(
        model_path.expanduser().resolve(), _model_settings(settings_path)
    )
    try:
        value = int(configured.get("mtp_num_draft_tokens") or fallback)
    except (TypeError, ValueError):
        value = fallback
    return value if value > 0 else fallback


def plans_for_models(
    models: list[LocalModel],
    settings_path: Path,
    helpers: list[LocalModel] | None = None,
) -> dict[Path, list[SpeculativePlan]]:
    settings = _model_settings(settings_path)
    fingerprints = {model.path: _tokenizer_fingerprint(model.path) for model in models}
    sizes = {model.path: _model_weight_bytes(model.path) for model in models}
    plans: dict[Path, list[SpeculativePlan]] = {}
    for target in models:
        options: list[SpeculativePlan] = []
        configured = _settings_for_model(target.path, settings)
        mtp_sidecar = find_mtp_sidecar(target.path)
        vlm_mtp_helper = _find_vlm_mtp_helper(target.path, helpers or [])
        if vlm_mtp_helper is not None:
            options.append(
                SpeculativePlan(
                    "mlx-vlm-mtp",
                    "assistant helper",
                    draft_model=vlm_mtp_helper.path,
                )
            )
        elif has_embedded_mtp(target.path):
            source = "OMLX config" if configured.get("mtp_enabled") else "detected"
            options.append(SpeculativePlan("omlx-mtp", source))
        elif mtp_sidecar is not None and _mtp_compatible(_read_config(target.path)):
            source = (
                "OMLX config + local MTP sidecar"
                if configured.get("mtp_enabled")
                else "local MTP sidecar"
            )
            options.append(
                SpeculativePlan(
                    "omlx-mtp",
                    source,
                    mtp_sidecar=mtp_sidecar,
                )
            )
        else:
            mtp_helper = _find_mtp_helper(target.path, helpers or [])
            if mtp_helper is not None:
                options.append(
                    SpeculativePlan(
                        "omlx-mtp",
                        "helper",
                        mtp_helper=mtp_helper.path,
                    )
                )

        fingerprint = fingerprints[target.path]
        target_size = sizes[target.path]
        compatible_drafts = [
            candidate
            for candidate in models
            if candidate.path != target.path
            and fingerprint is not None
            and fingerprints[candidate.path] == fingerprint
            and 0 < sizes[candidate.path] < target_size
            and _supports_external_draft(target.path)
            and _supports_external_draft(candidate.path)
        ]
        if compatible_drafts:
            draft = min(compatible_drafts, key=lambda item: sizes[item.path])
            options.append(SpeculativePlan("mlx-draft", "auto-detected", draft.path))
        plans[target.path] = options
    return plans


def choose_speculative_plan(
    target_path: Path,
    models: list[LocalModel],
    settings_path: Path,
    *,
    backend: str = "auto",
    draft_model: Path | None = None,
    helpers: list[LocalModel] | None = None,
) -> SpeculativePlan | None:
    if draft_model is not None:
        resolved_draft = draft_model.expanduser().resolve()
        if backend in ("auto", "mlx-vlm-mtp") and _is_vlm_mtp_pair(
            target_path, resolved_draft
        ):
            return SpeculativePlan(
                "mlx-vlm-mtp", "explicit", draft_model=resolved_draft
            )
        if backend == "mlx-vlm-mtp":
            return None
        if not (
            _supports_external_draft(target_path)
            and _supports_external_draft(resolved_draft)
        ):
            return None
        return SpeculativePlan("mlx-draft", "explicit", resolved_draft)
    all_plans = plans_for_models(models, settings_path, helpers)
    resolved_target = target_path.resolve()
    options = next(
        (
            candidates
            for model_path, candidates in all_plans.items()
            if model_path.resolve() == resolved_target
        ),
        [],
    )
    if backend == "auto":
        return options[0] if options else None
    return next((plan for plan in options if plan.backend == backend), None)
