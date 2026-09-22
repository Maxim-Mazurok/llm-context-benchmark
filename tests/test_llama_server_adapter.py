from __future__ import annotations

import json

from llm_context_benchmark.adapters.llama_server import LlamaServerAdapter


class FakeResponse:
    def __init__(self, events: list[dict[str, object]]) -> None:
        self.lines = [f"data: {json.dumps(event)}\n".encode() for event in events]

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)


def test_append_and_decode_preserves_sequence_and_streams_tokens(monkeypatch):
    adapter = LlamaServerAdapter("test-model")
    adapter._sequence_tokens = [1, 2]
    response = FakeResponse(
        [
            {"tokens": [5]},
            {"tokens": [6]},
            {"stop": True, "timings": {"cache_n": 2}},
        ]
    )
    requests: list[tuple[str, dict[str, object], bool]] = []

    def fake_request(endpoint, payload=None, *, stream=False):
        requests.append((endpoint, payload, stream))
        return response

    monkeypatch.setattr(adapter, "_request", fake_request)
    transitions: list[str] = []
    streamed_tokens: list[int] = []

    result = adapter.append_and_decode(
        [3, 4],
        2,
        lambda: transitions.append("decode"),
        lambda token, index, timestamp: streamed_tokens.append(token),
    )

    assert requests == [
        (
            "/completion",
            {
                "prompt": [1, 2, 3, 4],
                "n_predict": 2,
                "cache_prompt": True,
                "return_tokens": True,
                "stream": True,
                "id_slot": 0,
                "temperature": 0,
            },
            True,
        )
    ]
    assert transitions == ["decode"]
    assert streamed_tokens == [5, 6]
    assert result.generated_tokens == [5, 6]
    assert result.cache_catchup_tokens == 0
    assert adapter._sequence_tokens == [1, 2, 3, 4, 5, 6]


def test_make_input_tokens_adds_server_special_prefix(monkeypatch):
    adapter = LlamaServerAdapter("test-model")
    adapter._seed_tokens = [10, 11]
    monkeypatch.setattr(
        adapter,
        "_json_request",
        lambda endpoint, payload=None: {"tokens": [1, 10, 11]},
    )

    assert adapter.make_input_tokens(5, first=True) == [1, 10, 11, 10, 11]
    assert adapter.make_input_tokens(5) == [10, 11, 10, 11, 10]