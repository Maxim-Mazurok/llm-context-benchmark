export function contextForMetric(point, metric) {
  if (metric.startsWith("task_")) return point.task_prompt_tokens;
  if (metric === "prefill_tps" || metric === "prefill_duration_s") {
    return point.prefill_context_tokens;
  }
  return point.decode_context_tokens ?? point.context_tokens;
}

export function chartPoint(point, metric, xMetric = "context_tokens") {
  const rawX = xMetric === "context_tokens" ? contextForMetric(point, metric) : point[xMetric];
  const rawY = point[metric];
  if (rawX == null || rawX === "" || rawY == null || rawY === "") return null;

  const x = Number(rawX);
  const y = Number(rawY);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
  return { x, y, cycle: point.cycle, decodeKind: point.decode_kind };
}

export function individualBenchmarksForModel(modelBenchmark, benchmarks) {
  const sourceIds = new Set(modelBenchmark.sourceIds || []);
  return benchmarks.filter((benchmark) => sourceIds.has(benchmark.id));
}
