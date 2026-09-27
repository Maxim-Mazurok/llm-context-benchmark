"""Run the repository's growing-context harness for the distributed pinned-host A/B.

Run in WSL: python3 scripts/pinned_host_benchmark.py LABEL OUTPUT_DIR.
Only experiment instrumentation lives here; scheduling/reporting are the normal harness.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_context_benchmark.adapters.llama_server import LlamaServerAdapter
from llm_context_benchmark.runner import BenchmarkConfig, BenchmarkRunner


class InstrumentedAdapter(LlamaServerAdapter):
    def __init__(self, label: str, output: Path):
        super().__init__(None, server_url="http://172.18.224.1:8080")
        self.label = label
        self.output = output
        self.final_event = {}
        self.cycle = 0

    def _request(self, endpoint, payload=None, *, stream=False):
        if endpoint == "/completion" and stream:
            payload = {**payload, "ignore_eos": True, "seed": 1234,
                       "cache_prompt": self.cycle > 1}
            prompt = payload["prompt"]
            self.prompt_hash = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        response = super()._request(endpoint, payload, stream=stream)
        if not stream:
            return response
        adapter = self

        class RecordingResponse:
            def __enter__(self):
                response.__enter__()
                return self

            def __exit__(self, *args):
                return response.__exit__(*args)

            def __iter__(self):
                for line in response:
                    if line.startswith(b"data: "):
                        text = line[6:].strip()
                        if text != b"[DONE]":
                            event = json.loads(text)
                            if event.get("error"):
                                raise RuntimeError(event["error"])
                            if event.get("stop"):
                                adapter.final_event = event
                    yield line

        return RecordingResponse()

    def append_and_decode(self, input_tokens, max_tokens, on_prefill_complete, on_token):
        self.cycle += 1
        self.final_event = {}
        result = super().append_and_decode(input_tokens, max_tokens, on_prefill_complete, on_token)
        timings = self.final_event.get("timings", {})
        if not timings.get("prompt_ms") or "prompt_n" not in timings:
            raise RuntimeError("Missing native prompt timing; cannot compare variants reliably")
        if len(result.generated_tokens) != max_tokens:
            raise RuntimeError(f"Expected {max_tokens} exact decode tokens, got {len(result.generated_tokens)}")
        result.prefill_finished = result.prefill_started + timings["prompt_ms"] / 1000
        result.cache_catchup_tokens = max(0, timings["prompt_n"] - len(input_tokens))
        record = {"cycle": self.cycle, "prompt_sha256": self.prompt_hash,
                  "generated_sha256": hashlib.sha256(json.dumps(result.generated_tokens).encode()).hexdigest(),
                  "appended_tokens": len(input_tokens), "timings": timings,
                  "server_settings": self.final_event.get("generation_settings"),
                  "stop_type": self.final_event.get("stop_type")}
        with (self.output / "native-timings.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        return result

    def metadata(self):
        return {**super().metadata(), "experiment": "distributed-pinned-host-ab",
                "variant": self.label, "prefill_timing_source": "llama.cpp timings.prompt_ms",
                "decode_timing_source": "exact streamed token arrival intervals",
                "memory_scope": "WSL benchmark client only; see Windows telemetry separately",
                "server_context": 100000, "tensor_split": "5,26,4", "gpu_layers": 35,
                "batch": 512, "ubatch": 128, "kv_type": "q8_0", "load_mode": "none"}


if __name__ == "__main__":
    label, directory = sys.argv[1:]
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    adapter = InstrumentedAdapter(label, output)
    config = BenchmarkConfig(chunk_tokens=4096, short_decode_tokens=128,
                             long_decode_tokens=512, long_decode_interval=16384,
                             max_context=32768, sample_interval_ms=1000)
    result = BenchmarkRunner(adapter, config, output).run()
    print(json.dumps(result, indent=2), flush=True)
    if result.get("max_tested_context_tokens") not in (None, 32768):
        raise SystemExit("Run stopped before 32K; inspect report")
