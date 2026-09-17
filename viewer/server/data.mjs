import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

const GIB = 1024 ** 3;

export const METRICS = {
  context_tokens: { label: "Actual context", unit: "tokens", group: "Context" },
  prefill_tps: { label: "Prefill speed", unit: "tok/s", group: "Speed", phase: "prefill" },
  decode_tps: { label: "Decode speed", unit: "tok/s", group: "Speed", phase: "decode" },
  prefill_duration_s: { label: "Prefill duration", unit: "seconds", group: "Timing" },
  decode_duration_s: { label: "Decode duration", unit: "seconds", group: "Timing" },
  mlx_active_gib: { label: "MLX active max", unit: "GiB", group: "Memory" },
  mlx_peak_gib: { label: "MLX phase peak", unit: "GiB", group: "Memory" },
  system_available_gib: { label: "System available", unit: "GiB", group: "Memory" },
  compressed_gib: { label: "Compressed memory", unit: "GiB", group: "Memory" },
  swap_growth_gib: { label: "Swap growth", unit: "GiB", group: "Memory" },
  swap_total_gib: { label: "Total swap used", unit: "GiB", group: "Memory" },
  swapout_gib: { label: "Swap written", unit: "GiB", group: "Memory" },
  rss_gib: { label: "Process RSS max", unit: "GiB", group: "Memory" },
  cumulative_input_tokens: { label: "Cumulative input", unit: "tokens", group: "Context" },
  cumulative_generated_tokens: { label: "Cumulative generated", unit: "tokens", group: "Context" },
  memory_pressure: { label: "Memory pressure", unit: "level", group: "Memory" },
  task_prompt_tokens: { label: "Useful-task prompt", unit: "tokens", group: "Useful tasks" },
  task_generation_tps: { label: "Useful-task generation speed", unit: "tok/s", group: "Useful tasks" },
  task_prompt_tps: { label: "Useful-task prompt speed", unit: "tok/s", group: "Useful tasks" },
  task_ttft_s: { label: "Useful-task TTFT", unit: "seconds", group: "Useful tasks" },
  task_total_time_s: { label: "Useful-task total time", unit: "seconds", group: "Useful tasks" },
  task_score: { label: "Useful-task score", unit: "score", group: "Useful tasks" },
  task_mlx_active_gib: { label: "Useful-task MLX active peak", unit: "GiB", group: "Useful tasks" },
  task_footprint_gib: { label: "Useful-task footprint peak", unit: "GiB", group: "Useful tasks" },
};

async function readJson(file, fallback = null) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return fallback;
  }
}

function modelName(model) {
  const clean = String(model || "Unknown model").replace(/\/+$/, "");
  return clean.split("/").at(-1) || clean;
}

function benchmarkName(model, metadata) {
  const base = modelName(model);
  if (metadata?.decode_mode !== "speculative") return base;
  if (metadata?.speculative_backend === "omlx-mtp") return `${base} · MTP`;
  const draft = modelName(metadata?.draft_model || "draft");
  return `${base} · speculative via ${draft}`;
}

function localRunTime(id, metadata, fallbackMtime) {
  const match = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:--|$)/.exec(id);
  if (match) {
    const [, year, month, day, hour, minute, second] = match;
    const local = new Date(+year, +month - 1, +day, +hour, +minute, +second);
    return { epoch: local.getTime(), iso: `${year}-${month}-${day}T${hour}:${minute}:${second}` };
  }
  const raw = metadata?.started_or_finished_at;
  const parsed = raw ? new Date(raw) : fallbackMtime;
  return { epoch: parsed.getTime(), iso: parsed.toISOString() };
}

function maxNumber(values) {
  const finite = values.filter((value) => Number.isFinite(value));
  return finite.length ? Math.max(...finite) : null;
}

function pressureNumber(value) {
  return { normal: 0, warning: 1, critical: 2 }[value] ?? null;
}

function bytesToGib(value) {
  if (value === null || value === undefined || value === "") return null;
  return Number.isFinite(Number(value)) ? Number(value) / GIB : null;
}

function baseSummary(id, summary, metadata, phases, mtime, state = {}) {
  const model = metadata?.model || "Unknown model";
  const runTime = localRunTime(id, metadata, mtime);
  const initialSwap = Number(summary?.initial_swap_used_bytes || 0);
  const decodes = phases.filter((phase) => phase.phase === "decode");
  const finalDecode = decodes.at(-1);
  return {
    id,
    kind: "capacity-run",
    model,
    modelName: benchmarkName(model, metadata),
    runAt: runTime.iso,
    runAtEpoch: runTime.epoch,
    maxContextTokens: Number(summary?.max_tested_context_tokens || 0),
    practicalContextTokens: summary?.practical_context_tokens ?? null,
    hardLimitTokens: summary?.hard_model_runtime_limit_tokens ?? null,
    stopReason: summary?.stop_reason || state?.stop_reason || state?.status || "unknown",
    status: state?.status || metadata?.status || (summary?.stop_reason === "running" ? "running" : "completed"),
    resumable: Boolean(state?.resumable ?? metadata?.resumable)
      && !(summary?.hard_model_runtime_limit_tokens && Number(summary?.max_tested_context_tokens || 0) >= Number(summary.hard_model_runtime_limit_tokens)),
    baselineDecodeTps: summary?.baseline_decode_tokens_per_second ?? null,
    finalDecodeTps: finalDecode?.tokens_per_second ?? null,
    peakMlxActiveGib: bytesToGib(maxNumber(phases.map((phase) => phase.mlx_active_max_bytes))),
    peakSwapGrowthGib: bytesToGib(summary?.peak_swap_growth_bytes ?? maxNumber(
      phases.map((phase) => phase.swap_growth_peak_bytes),
    ) ?? Math.max(0, (maxNumber(phases.map((phase) => phase.swap_used_end_bytes)) || initialSwap) - initialSwap)),
    peakSwapUsedGib: bytesToGib(summary?.peak_swap_used_bytes ?? maxNumber(phases.map((phase) => phase.swap_used_peak_bytes))),
    swapoutGib: bytesToGib(summary?.swapout_bytes),
    worstMemoryPressure: summary?.worst_memory_pressure ?? null,
    phaseCount: phases.length,
  };
}

function usefulTaskSummary(id, summary, observations, mtime) {
  const model = summary?.models?.length === 1
    ? summary.models[0]
    : observations[0]?.model || "Mixed models";
  const runAt = summary?.created_at ? new Date(summary.created_at) : mtime;
  return {
    id,
    kind: "useful-task-run",
    model,
    modelName: modelName(model),
    runAt: runAt.toISOString(),
    runAtEpoch: runAt.getTime(),
    maxContextTokens: Number(summary?.max_prompt_tokens || 0),
    practicalContextTokens: null,
    hardLimitTokens: null,
    stopReason: "useful_tasks_imported",
    status: "completed",
    resumable: false,
    baselineDecodeTps: null,
    finalDecodeTps: null,
    peakMlxActiveGib: bytesToGib(maxNumber(
      observations.map((observation) => observation.mlx_active_peak_bytes),
    )),
    peakSwapGrowthGib: null,
    peakSwapUsedGib: null,
    swapoutGib: null,
    worstMemoryPressure: null,
    phaseCount: 0,
    observationCount: Number(summary?.observation_count || observations.length),
  };
}

function modelAggregateId(modelDisplayName) {
  return `model-${Buffer.from(modelDisplayName).toString("base64url")}`;
}

function mean(values) {
  const finite = values
    .filter((value) => value != null && value !== "" && Number.isFinite(Number(value)))
    .map(Number);
  return finite.length
    ? finite.reduce((total, value) => total + value, 0) / finite.length
    : null;
}

export function buildModelBenchmarks(benchmarks) {
  const groups = new Map();
  for (const benchmark of benchmarks) {
    const key = benchmark.modelName;
    const group = groups.get(key) || [];
    group.push(benchmark);
    groups.set(key, group);
  }
  return [...groups.entries()].map(([modelDisplayName, runs]) => {
    const latest = runs.reduce((current, run) =>
      run.runAtEpoch > current.runAtEpoch ? run : current,
    );
    const capacityRuns = runs.filter((run) => run.kind === "capacity-run");
    const usefulTaskRuns = runs.filter((run) => run.kind === "useful-task-run");
    return {
      id: modelAggregateId(modelDisplayName),
      kind: "model",
      model: latest.model,
      modelName: modelDisplayName,
      runAt: latest.runAt,
      runAtEpoch: latest.runAtEpoch,
      maxContextTokens: maxNumber(runs.map((run) => run.maxContextTokens)) || 0,
      practicalContextTokens: mean(capacityRuns.map((run) => run.practicalContextTokens)),
      hardLimitTokens: maxNumber(capacityRuns.map((run) => run.hardLimitTokens)),
      stopReason: "aggregate",
      status: runs.some((run) => ["running", "stopping"].includes(run.status))
        ? "running"
        : "completed",
      resumable: false,
      baselineDecodeTps: mean(capacityRuns.map((run) => run.baselineDecodeTps)),
      finalDecodeTps: mean(capacityRuns.map((run) => run.finalDecodeTps)),
      peakMlxActiveGib: mean(runs.map((run) => run.peakMlxActiveGib)),
      peakSwapGrowthGib: mean(capacityRuns.map((run) => run.peakSwapGrowthGib)),
      peakSwapUsedGib: mean(capacityRuns.map((run) => run.peakSwapUsedGib)),
      swapoutGib: mean(capacityRuns.map((run) => run.swapoutGib)),
      worstMemoryPressure: null,
      phaseCount: capacityRuns.reduce((total, run) => total + run.phaseCount, 0),
      runCount: capacityRuns.length,
      usefulTaskRunCount: usefulTaskRuns.length,
      observationCount: usefulTaskRuns.reduce(
        (total, run) => total + (run.observationCount || 0),
        0,
      ),
      sourceIds: runs.map((run) => run.id),
    };
  }).sort((left, right) => left.modelName.localeCompare(right.modelName));
}

export async function listBenchmarks(runsDir) {
  let entries = [];
  try {
    entries = await readdir(runsDir, { withFileTypes: true });
  } catch {
    return [];
  }
  const benchmarks = await Promise.all(
    entries.filter((entry) => entry.isDirectory()).map(async (entry) => {
      const directory = path.join(runsDir, entry.name);
      const [summary, metadata, phases, state, usefulSummary, observations, info] = await Promise.all([
        readJson(path.join(directory, "summary.json")),
        readJson(path.join(directory, "run-metadata.json"), {}),
        readJson(path.join(directory, "phases.json"), []),
        readJson(path.join(directory, "run-state.json"), {}),
        readJson(path.join(directory, "useful-task-summary.json")),
        readJson(path.join(directory, "useful-tasks.json"), []),
        stat(directory),
      ]);
      if (usefulSummary) {
        return usefulTaskSummary(entry.name, usefulSummary, observations, info.mtime);
      }
      if (!summary && !state?.status) return null;
      return baseSummary(entry.name, summary || {}, metadata, phases, info.mtime, state);
    }),
  );
  return benchmarks.filter(Boolean).sort((a, b) => b.runAtEpoch - a.runAtEpoch);
}

function usefulTaskPoints(observations) {
  return observations.map((observation, index) => ({
    cycle: index + 1,
    point_kind: "useful-task",
    context_tokens: observation.prompt_tokens,
    task_prompt_tokens: observation.prompt_tokens,
    task_generation_tps: observation.generation_tokens_per_second,
    task_prompt_tps: observation.prompt_tokens_per_second,
    task_ttft_s: observation.time_to_first_token_seconds,
    task_total_time_s: observation.total_time_seconds ?? observation.request_duration_seconds,
    task_score: observation.score,
    task_mlx_active_gib: bytesToGib(observation.mlx_active_peak_bytes),
    task_footprint_gib: bytesToGib(observation.physical_footprint_peak_bytes),
    task_id: observation.task_id,
    attempt_id: observation.attempt_id,
    sample_count: 1,
  }));
}

function averagePointGroup(points) {
  const numericKeys = new Set(points.flatMap((point) =>
    Object.entries(point)
      .filter(([, value]) => Number.isFinite(Number(value)))
      .map(([key]) => key),
  ));
  const averaged = {};
  for (const key of numericKeys) {
    averaged[key] = mean(points.map((point) => point[key]));
  }
  return {
    ...averaged,
    point_kind: points[0].point_kind,
    decode_kind: points[0].decode_kind,
    sample_count: points.length,
  };
}

function capacityMeasurementPoints(point) {
  const prefillContextTokens = point.prefill_context_tokens ?? point.context_tokens;
  const decodeContextTokens = point.decode_context_tokens ?? point.context_tokens;
  return [
    {
      point_kind: "capacity-prefill",
      context_tokens: prefillContextTokens,
      prefill_context_tokens: prefillContextTokens,
      prefill_tps: point.prefill_tps,
      prefill_duration_s: point.prefill_duration_s,
      cycle: point.cycle,
    },
    {
      point_kind: "capacity-decode",
      context_tokens: decodeContextTokens,
      decode_context_tokens: decodeContextTokens,
      decode_tps: point.decode_tps,
      decode_duration_s: point.decode_duration_s,
      decode_kind: point.decode_kind,
      cycle: point.cycle,
    },
    {
      ...point,
      point_kind: "capacity-memory",
      prefill_tps: null,
      prefill_duration_s: null,
      decode_tps: null,
      decode_duration_s: null,
    },
  ];
}

export function aggregateBenchmarkDetails(modelBenchmark, details) {
  const groupedPoints = new Map();
  for (const detail of details) {
    for (const point of detail.points || []) {
      const measurementPoints = point.point_kind === "useful-task"
        ? [point]
        : capacityMeasurementPoints(point);
      for (const measurementPoint of measurementPoints) {
        const pointKind = measurementPoint.point_kind;
      const context = pointKind === "useful-task"
          ? measurementPoint.task_prompt_tokens
          : measurementPoint.context_tokens;
      if (!Number.isFinite(Number(context))) continue;
        const decodeKind = pointKind === "capacity-decode"
          ? measurementPoint.decode_kind
          : "";
        const key = `${pointKind}:${context}:${decodeKind}`;
      const group = groupedPoints.get(key) || [];
        group.push(measurementPoint);
      groupedPoints.set(key, group);
      }
    }
  }
  const points = [...groupedPoints.values()]
    .map(averagePointGroup)
    .sort((left, right) => Number(left.context_tokens) - Number(right.context_tokens));
  return {
    ...modelBenchmark,
    points,
    metrics: METRICS,
    summary: {
      source_count: details.length,
      capacity_run_count: modelBenchmark.runCount,
      useful_task_run_count: modelBenchmark.usefulTaskRunCount,
      useful_task_observation_count: modelBenchmark.observationCount,
    },
    metadata: { aggregate: true, source_ids: modelBenchmark.sourceIds },
    measurement: {
      prefill: "Values at matching actual contexts are arithmetic means across capacity runs.",
      decode: "Values at matching actual contexts are arithmetic means across capacity runs.",
      memory: "Memory values at matching contexts are arithmetic means across source runs.",
      usefulTasks: "Useful-task values at matching prompt lengths are arithmetic means across imported observations.",
    },
  };
}

async function readSamples(directory) {
  try {
    const text = await readFile(path.join(directory, "samples.jsonl"), "utf8");
    return text.split("\n").filter(Boolean).map((line) => JSON.parse(line));
  } catch {
    return [];
  }
}

export function buildCyclePoints(phases, samples, initialSwap = 0) {
  const byPhase = new Map();
  for (const sample of samples) {
    byPhase.set(`${sample.phase_id}:${sample.phase}`, sample);
  }
  const decodeById = new Map(
    phases.filter((phase) => phase.phase === "decode").map((phase) => [phase.phase_id, phase]),
  );
  const points = [];
  for (const prefill of phases.filter((phase) => phase.phase === "prefill")) {
    const decode = decodeById.get(prefill.phase_id + 1);
    if (!decode) continue;
    const lastSample = byPhase.get(`${decode.phase_id}:decode`);
    const active = maxNumber([prefill.mlx_active_max_bytes, decode.mlx_active_max_bytes]);
    const peak = maxNumber([prefill.mlx_peak_max_bytes, decode.mlx_peak_max_bytes]);
    const rss = maxNumber([prefill.rss_max_bytes, decode.rss_max_bytes]);
    const swapEnd = Number(decode.swap_used_end_bytes ?? prefill.swap_used_end_bytes ?? initialSwap);
    const swapGrowthPeak = maxNumber([
      prefill.swap_growth_peak_bytes,
      decode.swap_growth_peak_bytes,
    ]);
    const swapUsedPeak = maxNumber([
      prefill.swap_used_peak_bytes,
      decode.swap_used_peak_bytes,
      swapEnd,
    ]);
    points.push({
      cycle: points.length + 1,
      context_tokens: Number(decode.context_end_tokens),
      prefill_context_tokens: Number(prefill.context_end_tokens),
      decode_context_tokens: Number(decode.context_end_tokens),
      prefill_tps: prefill.tokens_per_second ?? null,
      decode_tps: decode.tokens_per_second ?? null,
      decode_kind: decode.decode_kind || "short",
      prefill_duration_s: prefill.duration_s ?? null,
      decode_duration_s: decode.duration_s ?? null,
      mlx_active_gib: bytesToGib(active),
      mlx_peak_gib: bytesToGib(peak),
      system_available_gib: bytesToGib(
        decode.system_available_min_bytes ?? prefill.system_available_min_bytes ?? lastSample?.system_available_bytes,
      ),
      compressed_gib: bytesToGib(decode.compressed_end_bytes),
      swap_growth_gib: bytesToGib(Math.max(0, swapGrowthPeak ?? (swapEnd - initialSwap))),
      swap_total_gib: bytesToGib(swapUsedPeak),
      swapout_gib: bytesToGib(
        Math.max(0, Number(prefill.swapout_delta_bytes || 0)) + Math.max(0, Number(decode.swapout_delta_bytes || 0)),
      ),
      rss_gib: bytesToGib(rss),
      cumulative_input_tokens: decode.cumulative_input_tokens ?? null,
      cumulative_generated_tokens: decode.cumulative_generated_tokens ?? null,
      memory_pressure: pressureNumber(decode.pressure_worst),
    });
  }
  return points;
}

async function getIndividualBenchmark(runsDir, id) {
  if (!id || path.basename(id) !== id) return null;
  const directory = path.join(runsDir, id);
  const [summary, metadata, phases, state, info] = await Promise.all([
    readJson(path.join(directory, "summary.json")),
    readJson(path.join(directory, "run-metadata.json"), {}),
    readJson(path.join(directory, "phases.json"), []),
    readJson(path.join(directory, "run-state.json"), {}),
    stat(directory).catch(() => null),
  ]);
  if (!info) return null;
  if (!summary) {
    const [usefulSummary, observations] = await Promise.all([
      readJson(path.join(directory, "useful-task-summary.json")),
      readJson(path.join(directory, "useful-tasks.json"), []),
    ]);
    if (!usefulSummary) return null;
    return {
      ...usefulTaskSummary(id, usefulSummary, observations, info.mtime),
      summary: usefulSummary,
      metadata: { useful_tasks: true },
      points: usefulTaskPoints(observations),
      metrics: METRICS,
      measurement: {
        usefulTasks: "Each point is one imported independent useful-task request.",
      },
    };
  }
  const samples = phases.some((phase) => phase.system_available_min_bytes == null)
    ? await readSamples(directory)
    : [];
  const benchmark = baseSummary(id, summary, metadata, phases, info.mtime, state);
  const points = buildCyclePoints(phases, samples, Number(summary.initial_swap_used_bytes || 0));
  const legacyMemoryLayout = Number(metadata?.measurement_schema_version || 1) < 2;
  if (legacyMemoryLayout) {
    for (const point of points) {
      point.compressed_gib = null;
      point.swapout_gib = null;
    }
  }
  return {
    ...benchmark,
    summary,
    metadata,
    points,
    metrics: METRICS,
    measurement: {
      prefill: "Each point covers only that append phase, including its one-token cache catch-up when present.",
      decode: "Each point excludes time-to-first-token and covers only that decode probe.",
      memory: legacyMemoryLayout
        ? "Historical compressed-memory counters are hidden because those runs predate the corrected native macOS structure. Swap, pressure, MLX, and available-memory data remain valid."
        : "Swap growth uses the highest sampled value since model load; swap written is cumulative paging traffic during each phase.",
    },
  };
}

export async function getBenchmark(runsDir, id) {
  if (!id || path.basename(id) !== id) return null;
  if (!id.startsWith("model-")) return getIndividualBenchmark(runsDir, id);
  const benchmarks = await listBenchmarks(runsDir);
  const modelBenchmark = buildModelBenchmarks(benchmarks).find(
    (benchmark) => benchmark.id === id,
  );
  if (!modelBenchmark) return null;
  const details = (await Promise.all(
    modelBenchmark.sourceIds.map((sourceId) => getIndividualBenchmark(runsDir, sourceId)),
  )).filter(Boolean);
  return aggregateBenchmarkDetails(modelBenchmark, details);
}
