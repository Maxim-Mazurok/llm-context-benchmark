from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast


def emit(event: dict[str, object]) -> None:
    print(json.dumps(event, ensure_ascii=False), flush=True)


def build_prompt(
    tokenizer, instruction: str, document: str, reasoning_enabled: bool = False
) -> list[int]:
    content = f"{instruction.strip()}\n\n<document>\n{document}\n</document>"
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=reasoning_enabled,
        )
    else:
        tokens = tokenizer.encode(content)
    return [int(token) for token in tokens]


def response_sections(text: str) -> tuple[str, str, str]:
    thought_marker = "<|channel>thought"
    channel_end = "<channel|>"
    known_prefixes = (thought_marker, "<|channel>final")
    if thought_marker in text:
        thought = text.split(thought_marker, 1)[1]
        if channel_end not in thought:
            return thought.lstrip("\n"), "", "reasoning"
        reasoning, text = thought.split(channel_end, 1)
        reasoning = reasoning.lstrip("\n")
    else:
        reasoning = ""
    if text.startswith("<") and any(marker.startswith(text) for marker in known_prefixes):
        return reasoning, "", "protocol"
    for marker in ("<|channel>final\n", "<|channel>final", "<turn|>", "<eos>"):
        text = text.replace(marker, "")
    answer = text.lstrip("\n")
    return reasoning, answer, "answer" if answer else "protocol"


def model_context_limit(model, tokenizer) -> int:
    candidates: list[int] = []
    for source in (getattr(model, "args", None), getattr(model, "config", None)):
        for nested_source in (source, getattr(source, "text_config", None)):
            if nested_source is None:
                continue
            for name in ("max_position_embeddings", "max_seq_len", "model_max_length"):
                value = getattr(nested_source, name, None)
                if isinstance(value, int) and 0 < value < 10**9:
                    candidates.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
        candidates.append(tokenizer_limit)
    return min(candidates) if candidates else 1_000_000


def tokens_per_second(processed_tokens: int, elapsed_seconds: float) -> float | None:
    if processed_tokens <= 0 or elapsed_seconds <= 0:
        return None
    return processed_tokens / elapsed_seconds


def remaining_eta(
    limit: int | None, processed_tokens: int, speed: float | None
) -> float | None:
    if limit is None or speed is None or speed <= 0:
        return None
    return max(0, limit - processed_tokens) / speed


def run(
    model_path: Path,
    document_path: Path,
    instruction: str,
    max_output_tokens: int | None,
    reasoning_enabled: bool,
    max_reasoning_tokens: int | None,
) -> None:
    import mlx.core as mlx_core
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache

    mlx_core.set_wired_limit(0)
    emit({"type": "status", "phase": "loading"})
    loaded = cast(tuple[Any, ...], load(str(model_path)))
    model, tokenizer = loaded[0], loaded[1]
    mlx_core.set_wired_limit(0)

    document = document_path.read_text(encoding="utf-8", errors="replace")
    prompt_tokens = build_prompt(tokenizer, instruction, document, reasoning_enabled)
    del document
    emit({"type": "prompt", "totalTokens": len(prompt_tokens)})

    prompt_cache = make_prompt_cache(model)
    detokenizer = tokenizer.detokenizer
    prefill_started = time.perf_counter()
    prefill_finished: float | None = None

    def report_prefill(processed_tokens: int, total_tokens: int) -> None:
        nonlocal prefill_finished
        now = time.perf_counter()
        elapsed_seconds = max(now - prefill_started, 1e-9)
        current_tokens_per_second = tokens_per_second(processed_tokens, elapsed_seconds)
        emit(
            {
                "type": "prefill",
                "processedTokens": processed_tokens,
                "totalTokens": total_tokens,
                "tokensPerSecond": current_tokens_per_second,
                "elapsedSeconds": elapsed_seconds,
                "etaSeconds": remaining_eta(
                    total_tokens, processed_tokens, current_tokens_per_second
                ),
            }
        )
        if processed_tokens == total_tokens:
            prefill_finished = now

    maximum_generation_tokens = max(
        1, model_context_limit(model, tokenizer) - len(prompt_tokens)
    )
    stream = generate_step(
        mlx_core.array(prompt_tokens, dtype=mlx_core.int32),
        model,
        max_tokens=maximum_generation_tokens,
        prompt_cache=prompt_cache,
        prefill_step_size=512,
        prompt_progress_callback=report_prefill,
    )
    first_token_at: float | None = None
    generated_tokens = 0
    reasoning_tokens = 0
    output_tokens = 0
    reasoning_length = 0
    answer_length = 0
    finish_reason = "model_stop"
    end_token_ids = set(getattr(tokenizer, "eos_token_ids", []))
    end_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(end_token_id, int):
        end_token_ids.add(end_token_id)
    for token, _log_probabilities in stream:
        if int(token) in end_token_ids:
            break
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
            if prefill_finished is None:
                report_prefill(len(prompt_tokens), len(prompt_tokens))
        generated_tokens += 1
        detokenizer.add_token(int(token))
        reasoning, answer, stage = response_sections(detokenizer.text)
        reasoning_segment = reasoning[reasoning_length:]
        answer_segment = answer[answer_length:]
        reasoning_length = len(reasoning)
        answer_length = len(answer)
        if stage == "reasoning":
            reasoning_tokens += 1
        elif stage == "answer":
            output_tokens += 1
        elapsed_seconds = max(now - first_token_at, 0)
        measured_tokens = generated_tokens - 1
        current_tokens_per_second = tokens_per_second(measured_tokens, elapsed_seconds)
        active_limit = max_reasoning_tokens if stage == "reasoning" else max_output_tokens
        active_tokens = reasoning_tokens if stage == "reasoning" else output_tokens
        eta_seconds = remaining_eta(active_limit, active_tokens, current_tokens_per_second)
        emit(
            {
                "type": "decode",
                "stage": stage,
                "reasoning": reasoning_segment,
                "token": answer_segment,
                "generatedTokens": generated_tokens,
                "reasoningTokens": reasoning_tokens,
                "outputTokens": output_tokens,
                "tokensPerSecond": current_tokens_per_second,
                "elapsedSeconds": elapsed_seconds,
                "etaSeconds": eta_seconds,
            }
        )
        if stage == "reasoning" and max_reasoning_tokens is not None and reasoning_tokens >= max_reasoning_tokens:
            finish_reason = "reasoning_limit"
            break
        if stage == "answer" and max_output_tokens is not None and output_tokens >= max_output_tokens:
            finish_reason = "output_limit"
            break

    detokenizer.finalize()
    reasoning, answer, _stage = response_sections(detokenizer.text)
    final_reasoning_segment = reasoning[reasoning_length:]
    final_answer_segment = answer[answer_length:]
    if final_reasoning_segment or final_answer_segment:
        emit({"type": "text", "reasoning": final_reasoning_segment, "token": final_answer_segment})
    emit(
        {
            "type": "complete",
            "generatedTokens": generated_tokens,
            "reasoningTokens": reasoning_tokens,
            "outputTokens": output_tokens,
            "finishReason": finish_reason,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--reasoning", action="store_true")
    parser.add_argument("--max-reasoning-tokens", type=int)
    arguments = parser.parse_args()
    try:
        run(
            arguments.model.expanduser(),
            arguments.document,
            arguments.prompt,
            arguments.max_output_tokens,
            arguments.reasoning,
            arguments.max_reasoning_tokens,
        )
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())