from __future__ import annotations

import importlib.metadata
import json
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from .base import CycleResult


class MLXLMAdapter:
    """Direct mlx-lm adapter that keeps one prompt cache for the whole run.

    mlx-lm's ``generate_step`` cache trails the returned stream by one token.
    We retain that last token and prepend it to the next input chunk. This is
    the key detail that makes separate measurement cycles one continuous
    sequence rather than independent requests.
    """

    name = "mlx-lm"

    def __init__(
        self,
        model: str,
        *,
        speculative_backend: str | None = None,
        draft_model: str | None = None,
        mtp_sidecar: str | None = None,
        mtp_helper: str | None = None,
        num_draft_tokens: int = 2,
        prefill_step_size: int = 2048,
        kv_bits: int | None = None,
        kv_group_size: int = 64,
        seed_text: str | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model
        self.speculative_backend = speculative_backend
        self.draft_model_id = draft_model
        self.mtp_sidecar = mtp_sidecar
        self.mtp_helper = mtp_helper
        self.num_draft_tokens = num_draft_tokens
        self.prefill_step_size = prefill_step_size
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.seed_text = seed_text or (
            "The benchmark extends one continuous conversation with neutral prose, "
            "measurements, explanations, and repeated factual structure. "
        )
        self.trust_remote_code = trust_remote_code
        self.context_limit: int | None = None
        self.model = None
        self.draft_model = None
        self.tokenizer = None
        self.processor = None
        self.draft_tokenizer = None
        self.prompt_cache = None
        self._mtp_generator = None
        self._mtp_cached_tokens: list[int] = []
        self._mtp_pending_tokens: list[int] = []
        self._pending_token: int | None = None
        self._seed_tokens: list[int] = []
        self._versions: dict[str, str] = {}
        self._mtp_overlay: tempfile.TemporaryDirectory[str] | None = None

    def load(self) -> None:
        import mlx.core as mx

        if self.speculative_backend == "mlx-vlm-mtp":
            if not self.draft_model_id:
                raise ValueError("MLX-VLM MTP requires an assistant draft model")
            from mlx_vlm.speculative.drafters import load_drafter
            from mlx_vlm.utils import load as load_vlm

            self.model, self.processor = load_vlm(self.model_id)
            self.tokenizer = self.processor.tokenizer
            self.draft_model, draft_kind = load_drafter(
                self.draft_model_id, kind="mtp"
            )
            if draft_kind != "mtp":
                raise ValueError("MLX-VLM assistant did not resolve to MTP")
        else:
            from mlx_lm import load

            load_model_id = self.model_id
            if self.speculative_backend == "omlx-mtp":
                if self.kv_bits is not None:
                    raise ValueError("OMLX MTP does not support --kv-bits in this adapter")
                from omlx.model_settings import ModelSettings
                from omlx.utils.model_loading import maybe_apply_pre_load_patches

                if self.mtp_sidecar or self.mtp_helper:
                    load_model_id = self._prepare_mtp_overlay()
                settings = ModelSettings(
                    mtp_enabled=True,
                    mtp_num_draft_tokens=self.num_draft_tokens,
                )
                maybe_apply_pre_load_patches(load_model_id, model_settings=settings)

            load_arguments = {
                "tokenizer_config": {"trust_remote_code": self.trust_remote_code}
            }
            try:
                self.model, self.tokenizer = load(load_model_id, **load_arguments)
            except TypeError:
                self.model, self.tokenizer = load(load_model_id)
            if self.speculative_backend == "mlx-draft":
                if not self.draft_model_id:
                    raise ValueError("MLX draft speculation requires a draft model")
                try:
                    self.draft_model, self.draft_tokenizer = load(
                        self.draft_model_id, **load_arguments
                    )
                except TypeError:
                    self.draft_model, self.draft_tokenizer = load(self.draft_model_id)
                self._validate_draft_tokenizer()
        encoded = self.tokenizer.encode(self.seed_text, add_special_tokens=False)
        self._seed_tokens = [int(t) for t in encoded]
        if not self._seed_tokens:
            raise RuntimeError(
                "Tokenizer produced no tokens for the benchmark seed text"
            )
        bos = getattr(self.tokenizer, "bos_token_id", None)
        warmup = ([int(bos)] if isinstance(bos, int) else []) + self._seed_tokens[:31]
        if self.speculative_backend == "omlx-mtp":
            if not self._mtp_enabled_on_model():
                raise RuntimeError(
                    "OMLX MTP was requested, but the loaded target did not activate "
                    "an embedded MTP head"
                )
            self._run_mtp_warmup(warmup)
        elif self.speculative_backend == "mlx-vlm-mtp":
            list(self._vlm_mtp_stream(warmup, 4, prompt_cache=None))
        elif self.speculative_backend == "mlx-draft":
            from mlx_lm.models.cache import make_prompt_cache
            from mlx_lm.generate import speculative_generate_step

            warmup_cache = make_prompt_cache(self.model) + make_prompt_cache(
                self.draft_model
            )
            list(
                speculative_generate_step(
                    mx.array(warmup, dtype=mx.int32),
                    self.model,
                    self.draft_model,
                    num_draft_tokens=self.num_draft_tokens,
                    max_tokens=4,
                    prompt_cache=warmup_cache,
                    prefill_step_size=self.prefill_step_size,
                )
            )
        else:
            from mlx_lm.generate import generate_step
            from mlx_lm.models.cache import make_prompt_cache

            warmup_cache = make_prompt_cache(self.model)
            list(
                generate_step(
                    mx.array(warmup, dtype=mx.int32),
                    self.model,
                    max_tokens=4,
                    prompt_cache=warmup_cache,
                    prefill_step_size=self.prefill_step_size,
                )
            )
        mx.synchronize()
        mx.clear_cache()
        if self.speculative_backend == "omlx-mtp":
            self._mtp_generator = self._create_mtp_generator()
        elif self.speculative_backend == "mlx-vlm-mtp":
            from mlx_vlm.models.cache import make_prompt_cache

            self.prompt_cache = make_prompt_cache(self.model.language_model)
            mx.set_wired_limit(0)
        elif self.speculative_backend == "mlx-draft":
            from mlx_lm.models.cache import make_prompt_cache

            self.prompt_cache = make_prompt_cache(self.model) + make_prompt_cache(
                self.draft_model
            )
        else:
            from mlx_lm.models.cache import make_prompt_cache

            self.prompt_cache = make_prompt_cache(self.model)
        self.context_limit = self._detect_context_limit()
        if self.speculative_backend == "mlx-draft":
            draft_limit = self._detect_context_limit_for(
                self.draft_model, self.draft_tokenizer, self.draft_model_id
            )
            if draft_limit is not None:
                self.context_limit = (
                    min(self.context_limit, draft_limit)
                    if self.context_limit is not None
                    else draft_limit
                )
        for package in ("mlx", "mlx-lm", "mlx-vlm", "transformers"):
            try:
                self._versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        mx.reset_peak_memory()

    def _prepare_mtp_overlay(self) -> str:
        import mlx.core as mx

        source_root = Path(self.model_id).expanduser().resolve()
        sidecar = (
            Path(self.mtp_sidecar).expanduser().resolve()
            if self.mtp_sidecar
            else None
        )
        helper = (
            Path(self.mtp_helper).expanduser().resolve()
            if self.mtp_helper
            else None
        )
        if not source_root.is_dir():
            raise ValueError("MTP target directory does not exist")
        if sidecar is not None and not sidecar.is_file():
            raise ValueError("MTP sidecar does not exist")
        if helper is not None and not helper.is_dir():
            raise ValueError("MTP helper directory does not exist")
        self._mtp_overlay = tempfile.TemporaryDirectory(
            prefix="llm-context-benchmark-mtp-"
        )
        overlay = Path(self._mtp_overlay.name)
        for source in source_root.iterdir():
            if sidecar is not None and source.resolve() == sidecar:
                continue
            (overlay / source.name).symlink_to(
                source, target_is_directory=source.is_dir()
            )
        if sidecar is not None:
            (overlay / "model-mtp-sidecar.safetensors").symlink_to(sidecar)
        elif helper is not None:
            helper_weights = {}
            for shard in sorted(helper.glob("model*.safetensors")):
                helper_weights.update(mx.load(str(shard)))
            if not helper_weights:
                raise ValueError("MTP helper contains no model safetensors")
            prefixed = {
                key if key.startswith("mtp.") else f"mtp.{key}": value
                for key, value in helper_weights.items()
            }
            mx.save_safetensors(
                str(overlay / "model-mtp-helper.safetensors"),
                prefixed,
                metadata={"format": "mlx"},
            )
        return str(overlay)

    def _validate_draft_tokenizer(self) -> None:
        attributes = ("vocab_size", "bos_token_id", "eos_token_id")
        mismatches = [
            name
            for name in attributes
            if getattr(self.tokenizer, name, None)
            != getattr(self.draft_tokenizer, name, None)
        ]
        target_probe = self.tokenizer.encode(self.seed_text, add_special_tokens=False)
        draft_probe = self.draft_tokenizer.encode(
            self.seed_text, add_special_tokens=False
        )
        if mismatches or list(target_probe) != list(draft_probe):
            detail = ", ".join(mismatches) if mismatches else "seed tokenization"
            raise ValueError(
                f"Target and draft tokenizers are incompatible ({detail})"
            )

    def _mtp_enabled_on_model(self) -> bool:
        candidates = [self.model]
        seen: set[int] = set()
        while candidates:
            candidate = candidates.pop()
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            if getattr(candidate, "_omlx_mtp_decode_enabled", False):
                return True
            for name in ("model", "language_model", "text_model"):
                child = getattr(candidate, name, None)
                if child is not None:
                    candidates.append(child)
        return False

    def _run_mtp_warmup(self, prompt: list[int]) -> None:
        generator = self._create_mtp_generator()
        uid = generator.insert([prompt], max_tokens=[4])[0]
        finished = False
        while not finished:
            _prompt, generated = generator.next()
            finished = any(
                response.uid == uid and response.finish_reason is not None
                for response in generated
            )
        generator.close()

    def _create_mtp_generator(self):
        import mlx.core as mlx_core
        from mlx_lm.generate import BatchGenerator

        generator = BatchGenerator(
            self.model,
            prefill_step_size=self.prefill_step_size,
            completion_batch_size=1,
            prefill_batch_size=1,
        )
        mlx_core.set_wired_limit(0)
        return generator

    def _detect_context_limit(self) -> int | None:
        return self._detect_context_limit_for(
            self.model, self.tokenizer, self.model_id
        )

    @staticmethod
    def _detect_context_limit_for(model, tokenizer, model_id: str) -> int | None:
        candidates: list[int] = []
        config = getattr(model, "args", None)
        for source in (config, getattr(model, "config", None)):
            if source is None:
                continue
            nested_sources = (source, getattr(source, "text_config", None))
            for nested_source in nested_sources:
                if nested_source is None:
                    continue
                for key in (
                    "max_position_embeddings",
                    "max_seq_len",
                    "model_max_length",
                ):
                    value = getattr(nested_source, key, None)
                    if isinstance(value, int) and 0 < value < 10**9:
                        candidates.append(value)
        path = Path(model_id).expanduser()
        if path.is_dir() and (path / "config.json").exists():
            try:
                raw = json.loads((path / "config.json").read_text())
                raw_sources = (raw, raw.get("text_config"))
                for raw_source in raw_sources:
                    if not isinstance(raw_source, dict):
                        continue
                    for key in (
                        "max_position_embeddings",
                        "max_seq_len",
                        "model_max_length",
                    ):
                        value = raw_source.get(key)
                        if isinstance(value, int) and 0 < value < 10**9:
                            candidates.append(value)
            except (OSError, json.JSONDecodeError):
                pass
        tokenizer_limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
            candidates.append(tokenizer_limit)
        return min(candidates) if candidates else None

    def make_input_tokens(self, count: int, first: bool = False) -> list[int]:
        if count <= 0:
            return []
        tokens: list[int] = []
        if first:
            bos = getattr(self.tokenizer, "bos_token_id", None)
            if isinstance(bos, int):
                tokens.append(bos)
        needed = count - len(tokens)
        repeats = (needed + len(self._seed_tokens) - 1) // len(self._seed_tokens)
        tokens.extend((self._seed_tokens * repeats)[:needed])
        return tokens

    def restore_context(self, tokens: list[int]) -> None:
        """Rebuild the live cache from a completed-cycle token checkpoint."""
        if not tokens:
            return
        if len(tokens) == 1:
            self._pending_token = tokens[0]
            self._mtp_pending_tokens = list(tokens)
            return
        prompt, pending = tokens[:-1], tokens[-1]
        if self.speculative_backend == "omlx-mtp":
            uid = self._mtp_generator.insert([prompt], max_tokens=[1])[0]
            final_response = None
            while final_response is None:
                _prefill, generated = self._mtp_generator.next()
                final_response = next(
                    (
                        response
                        for response in generated
                        if response.uid == uid and response.finish_reason is not None
                    ),
                    None,
                )
            self.prompt_cache = final_response.prompt_cache
            cache_offset = min(self._cache_offset(self.prompt_cache), len(prompt))
            self._mtp_cached_tokens = prompt[:cache_offset]
            self._mtp_pending_tokens = [*prompt[cache_offset:], pending]
            return

        import mlx.core as mx

        if self.speculative_backend == "mlx-vlm-mtp":
            next(iter(self._vlm_mtp_stream(prompt, 1)), None)
        elif self.speculative_backend == "mlx-draft":
            from mlx_lm.generate import speculative_generate_step

            stream = speculative_generate_step(
                mx.array(prompt, dtype=mx.int32),
                self.model,
                self.draft_model,
                num_draft_tokens=self.num_draft_tokens,
                max_tokens=1,
                prompt_cache=self.prompt_cache,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
            )
        else:
            from mlx_lm.generate import generate_step

            stream = generate_step(
                mx.array(prompt, dtype=mx.int32),
                self.model,
                max_tokens=1,
                prompt_cache=self.prompt_cache,
                prefill_step_size=self.prefill_step_size,
                kv_bits=self.kv_bits,
                kv_group_size=self.kv_group_size,
            )
        next(iter(stream), None)
        mx.synchronize()
        self._pending_token = pending

    def append_and_decode(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        if self.speculative_backend == "omlx-mtp":
            return self._append_mtp(
                input_tokens, max_tokens, on_prefill_complete, on_token
            )
        if self.speculative_backend == "mlx-vlm-mtp":
            return self._append_vlm_mtp(
                input_tokens, max_tokens, on_prefill_complete, on_token
            )
        if self.speculative_backend == "mlx-draft":
            return self._append_external_draft(
                input_tokens, max_tokens, on_prefill_complete, on_token
            )

        import mlx.core as mx
        from mlx_lm.generate import generate_step

        catchup = 1 if self._pending_token is not None else 0
        prompt = (
            [self._pending_token] if self._pending_token is not None else []
        ) + input_tokens
        prompt_array = mx.array(prompt, dtype=mx.int32)
        prefill_started = time.perf_counter()
        prefill_finished: float | None = None

        def progress(processed: int, total: int) -> None:
            nonlocal prefill_finished
            if processed == total and prefill_finished is None:
                prefill_finished = time.perf_counter()
                on_prefill_complete()

        stream = generate_step(
            prompt_array,
            self.model,
            max_tokens=max_tokens,
            prompt_cache=self.prompt_cache,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
            prompt_progress_callback=progress,
        )
        generated: list[int] = []
        timestamps: list[float] = []
        for index, (token, _logprobs) in enumerate(stream):
            now = time.perf_counter()
            value = int(token)
            generated.append(value)
            timestamps.append(now)
            on_token(value, index, now)
        if prefill_finished is None:
            prefill_finished = time.perf_counter()
            on_prefill_complete()
        self._pending_token = generated[-1] if generated else self._pending_token
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=catchup,
            generated_tokens=generated,
            prefill_started=prefill_started,
            prefill_finished=prefill_finished,
            token_timestamps=timestamps,
        )

    def _append_external_draft(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        import mlx.core as mx
        from mlx_lm.generate import speculative_generate_step

        catchup = 1 if self._pending_token is not None else 0
        prompt = (
            [self._pending_token] if self._pending_token is not None else []
        ) + input_tokens
        prefill_started = time.perf_counter()
        prefill_finished: float | None = None
        stream = speculative_generate_step(
            mx.array(prompt, dtype=mx.int32),
            self.model,
            self.draft_model,
            num_draft_tokens=self.num_draft_tokens,
            max_tokens=max_tokens,
            prompt_cache=self.prompt_cache,
            prefill_step_size=self.prefill_step_size,
            kv_bits=self.kv_bits,
            kv_group_size=self.kv_group_size,
        )
        generated: list[int] = []
        timestamps: list[float] = []
        for index, (token, _logprobs, _from_draft) in enumerate(stream):
            now = time.perf_counter()
            if prefill_finished is None:
                prefill_finished = now
                on_prefill_complete()
            value = int(token)
            generated.append(value)
            timestamps.append(now)
            on_token(value, index, now)
        if prefill_finished is None:
            prefill_finished = time.perf_counter()
            on_prefill_complete()
        self._pending_token = generated[-1] if generated else self._pending_token
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=catchup,
            generated_tokens=generated,
            prefill_started=prefill_started,
            prefill_finished=prefill_finished,
            token_timestamps=timestamps,
        )

    def _vlm_mtp_stream(self, prompt: list[int], max_tokens: int, prompt_cache=None):
        import mlx.core as mx
        from mlx_vlm.generate import generate_step

        return generate_step(
            mx.array([prompt], dtype=mx.int32),
            self.model,
            None,
            None,
            max_tokens=max_tokens,
            temperature=0,
            prompt_cache=self.prompt_cache if prompt_cache is None else prompt_cache,
            prefill_step_size=self.prefill_step_size,
            draft_model=self.draft_model,
            draft_kind="mtp",
            draft_block_size=self.num_draft_tokens + 1,
        )

    def _append_vlm_mtp(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        catchup = 1 if self._pending_token is not None else 0
        prompt = (
            [self._pending_token] if self._pending_token is not None else []
        ) + input_tokens
        prefill_started = time.perf_counter()
        prefill_finished: float | None = None
        generated: list[int] = []
        timestamps: list[float] = []
        for index, (token, _logprobs) in enumerate(
            self._vlm_mtp_stream(prompt, max_tokens)
        ):
            now = time.perf_counter()
            if prefill_finished is None:
                prefill_finished = now
                on_prefill_complete()
            value = int(token)
            generated.append(value)
            timestamps.append(now)
            on_token(value, index, now)
        if prefill_finished is None:
            prefill_finished = time.perf_counter()
            on_prefill_complete()
        self._pending_token = generated[-1] if generated else self._pending_token
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=catchup,
            generated_tokens=generated,
            prefill_started=prefill_started,
            prefill_finished=prefill_finished,
            token_timestamps=timestamps,
        )

    @staticmethod
    def _cache_offset(prompt_cache) -> int:
        offsets = [
            int(offset)
            for item in prompt_cache or []
            if (offset := getattr(item, "offset", None)) is not None
        ]
        return min(offsets) if offsets else 0

    def _append_mtp(
        self,
        input_tokens: list[int],
        max_tokens: int,
        on_prefill_complete: Callable[[], None],
        on_token: Callable[[int, int, float], None],
    ) -> CycleResult:
        catchup = len(self._mtp_pending_tokens)
        prompt = [*self._mtp_pending_tokens, *input_tokens]
        self._mtp_pending_tokens = []
        caches = [self.prompt_cache] if self.prompt_cache is not None else None
        cached_tokens = [self._mtp_cached_tokens] if caches is not None else None
        uid = self._mtp_generator.insert(
            [prompt],
            max_tokens=[max_tokens],
            caches=caches,
            all_tokens=cached_tokens,
        )[0]
        prefill_started = time.perf_counter()
        prefill_finished: float | None = None
        generated: list[int] = []
        timestamps: list[float] = []
        final_response = None
        while final_response is None:
            prompt_responses, generation_responses = self._mtp_generator.next()
            if prefill_finished is None and any(
                response.uid == uid and response.end_of_prompt
                for response in prompt_responses
            ):
                prefill_finished = time.perf_counter()
                on_prefill_complete()
            for response in generation_responses:
                if response.uid != uid:
                    continue
                now = time.perf_counter()
                value = int(response.token)
                generated.append(value)
                timestamps.append(now)
                on_token(value, len(generated) - 1, now)
                if response.finish_reason is not None:
                    final_response = response
        if prefill_finished is None:
            prefill_finished = timestamps[0] if timestamps else time.perf_counter()
            on_prefill_complete()

        self.prompt_cache = final_response.prompt_cache
        all_tokens = [int(token) for token in (final_response.all_tokens or [])]
        cache_offset = min(self._cache_offset(self.prompt_cache), len(all_tokens))
        self._mtp_cached_tokens = all_tokens[:cache_offset]
        self._mtp_pending_tokens = all_tokens[cache_offset:]
        return CycleResult(
            appended_input_tokens=len(input_tokens),
            cache_catchup_tokens=catchup,
            generated_tokens=generated,
            prefill_started=prefill_started,
            prefill_finished=prefill_finished,
            token_timestamps=timestamps,
        )

    def mlx_metrics(self) -> tuple[int | None, int | None, int | None]:
        import mlx.core as mx

        return (
            int(mx.get_active_memory()),
            int(mx.get_peak_memory()),
            int(mx.get_cache_memory()),
        )

    def reset_peak_memory(self) -> None:
        import mlx.core as mx

        mx.reset_peak_memory()

    def metadata(self) -> dict[str, object]:
        return {
            "adapter": self.name,
            "model": self.model_id,
            "context_limit": self.context_limit,
            "versions": self._versions,
            "prefill_step_size": self.prefill_step_size,
            "kv_bits": self.kv_bits,
            "kv_group_size": self.kv_group_size,
            "continuous_cache": True,
            "warmup_tokens": 4,
            "decode_mode": (
                "speculative" if self.speculative_backend else "autoregressive"
            ),
            "speculative_backend": self.speculative_backend,
            "draft_model": self.draft_model_id,
            "mtp_sidecar": self.mtp_sidecar,
            "mtp_helper": self.mtp_helper,
            "num_draft_tokens": (
                self.num_draft_tokens if self.speculative_backend else None
            ),
        }
