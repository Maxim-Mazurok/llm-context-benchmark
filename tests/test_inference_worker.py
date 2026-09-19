from llm_context_benchmark.inference_worker import (
    build_prompt,
    remaining_eta,
    response_sections,
    tokens_per_second,
)


class ChatTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is True
        assert add_generation_prompt is True
        assert enable_thinking is False
        assert messages == [
            {
                "role": "user",
                "content": "Summarize this\n\n<document>\nDocument body\n</document>",
            }
        ]
        return [1, 2, 3]


def test_build_prompt_uses_chat_template():
    assert build_prompt(ChatTokenizer(), " Summarize this ", "Document body") == [1, 2, 3]


def test_response_sections_separates_reasoning_and_answer():
    assert response_sections("<|chan") == ("", "", "protocol")
    assert response_sections("<|channel>thought\nprivate") == ("private", "", "reasoning")
    assert response_sections("<|channel>thought\nprivate<channel|><|chan") == ("private", "", "protocol")
    assert response_sections("<|channel>thought\nprivate<channel|><|channel>final\nAnswer") == ("private", "Answer", "answer")


def test_response_sections_preserves_direct_answers():
    assert response_sections("Direct answer") == ("", "Direct answer", "answer")


def test_startup_telemetry_ignores_unmeasurable_samples():
    assert tokens_per_second(0, 0.001) is None
    assert tokens_per_second(1, 0) is None
    assert remaining_eta(100, 1, None) is None
    assert remaining_eta(None, 1, 20) is None


def test_telemetry_calculates_finite_eta():
    assert tokens_per_second(20, 2) == 10
    assert remaining_eta(100, 20, 10) == 8