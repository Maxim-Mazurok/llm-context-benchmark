from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalModel:
    path: Path
    relative_name: str
    exclusion_reason: str | None = None


_AUXILIARY_NAME_WORDS = {
    "assistant",
    "draft",
    "embed",
    "embedding",
    "embeddings",
    "mtp",
    "rerank",
    "reranker",
    "reward",
    "speculator",
}
_AUXILIARY_CONFIG_MARKERS = (
    "assistant",
    "embedding",
    "forfeatureextraction",
    "formaskedlm",
    "forsequenceclassification",
    "fortokenclassification",
    "mtp",
    "reranker",
    "reward",
    "speculator",
    "xlmrobertamodel",
)


def _classify_model(path: Path, config: dict[str, object]) -> str | None:
    name_words = set(re.split(r"[^a-z0-9]+", path.name.casefold()))
    matched_words = sorted(name_words & _AUXILIARY_NAME_WORDS)
    if matched_words:
        return f"auxiliary name marker: {matched_words[0]}"

    model_type = str(config.get("model_type") or "")
    architectures = config.get("architectures") or []
    if not isinstance(architectures, list):
        architectures = [architectures]
    config_identity = " ".join(
        [model_type, *(str(value) for value in architectures)]
    ).casefold()
    for marker in _AUXILIARY_CONFIG_MARKERS:
        if marker in config_identity:
            return f"auxiliary/non-generative config marker: {marker}"
    return None


def discover_omlx_models(root: Path) -> tuple[list[LocalModel], list[LocalModel]]:
    root = root.expanduser().resolve()
    included: list[LocalModel] = []
    excluded: list[LocalModel] = []
    if not root.is_dir():
        return included, excluded

    model_paths: list[Path] = []
    for provider_path in sorted(root.iterdir()):
        if provider_path.name.startswith(".") or not provider_path.is_dir():
            continue
        if (provider_path / "config.json").is_file():
            model_paths.append(provider_path)
        for model_path in sorted(provider_path.iterdir()):
            if model_path.name.startswith(".") or not model_path.is_dir():
                continue
            if (model_path / "config.json").is_file():
                model_paths.append(model_path)

    for model_path in model_paths:
        config_path = model_path / "config.json"
        relative = model_path.relative_to(root)
        relative_name = relative.as_posix()
        try:
            config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            excluded.append(
                LocalModel(
                    model_path,
                    relative_name,
                    f"invalid config: {type(exc).__name__}",
                )
            )
            continue
        has_weights = any(model_path.glob("*.safetensors")) or any(
            model_path.glob("*.npz")
        )
        if not has_weights:
            excluded.append(
                LocalModel(model_path, relative_name, "no local model weights")
            )
            continue
        reason = _classify_model(model_path, config)
        record = LocalModel(model_path, relative_name, reason)
        (excluded if reason else included).append(record)
    return included, excluded


def safe_model_slug(relative_name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "--", relative_name).strip("-.")
    return slug or "model"
