from __future__ import annotations

from llm_context_benchmark.native_metrics import PortableMetrics, make_native_metrics


def test_portable_metrics_reports_a_nonzero_peak_rss() -> None:
    peak = PortableMetrics().snapshot().process_peak_rss_bytes

    assert peak is not None
    assert peak > 0


def test_native_metrics_factory_is_available_on_the_current_platform() -> None:
    snapshot = make_native_metrics().snapshot()

    assert snapshot.process_peak_rss_bytes is not None
