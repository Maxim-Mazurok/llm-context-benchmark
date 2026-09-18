export function legendLabelMode(series, runsById) {
  const runIds = [...new Set(series.map((item) => item.benchmarkId))];
  if (runIds.length <= 1) return "metric";

  const metrics = new Set(series.map((item) => item.metric));
  if (runIds.every((id) => runsById[id]?.kind === "model")) {
    return metrics.size === 1 ? "model" : "model-metric";
  }

  const modelNames = runIds.map((id) => runsById[id]?.modelName).filter(Boolean);
  if (modelNames.length === runIds.length && new Set(modelNames).size === runIds.length) {
    return metrics.size === 1 ? "model" : "model-metric";
  }
  return "run";
}

export function seriesLabel(run, metric, mode) {
  if (mode === "metric") return metric.label;
  if (mode === "model") return run.modelName;
  if (mode === "model-metric" || run.kind === "model") return `${run.modelName} · ${metric.label}`;

  const runTime = new Date(run.runAt).toLocaleString(undefined, {
    dateStyle: "short",
    timeStyle: "short",
  });
  return `${run.modelName} · ${runTime} · ${metric.label}`;
}
