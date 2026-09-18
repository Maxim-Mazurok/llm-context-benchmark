import assert from "node:assert/strict";
import test from "node:test";

import {
  aggregateBenchmarkDetails,
  buildCyclePoints,
  buildModelBenchmarks,
  canonicalModelName,
} from "./data.mjs";

test("prefill throughput stays phase-local and long decode remains metadata", () => {
  const phases = [
    { phase_id: 1, phase: "prefill", context_end_tokens: 5000, tokens_per_second: 1000, duration_s: 5 },
    { phase_id: 2, phase: "decode", decode_kind: "short", context_end_tokens: 5256, tokens_per_second: 100, duration_s: 2 },
    { phase_id: 3, phase: "prefill", context_end_tokens: 10256, tokens_per_second: 500, duration_s: 10 },
    { phase_id: 4, phase: "decode", decode_kind: "long", context_end_tokens: 11256, tokens_per_second: 80, duration_s: 12 },
  ];
  const points = buildCyclePoints(phases, [], 0);
  assert.deepEqual(points.map((point) => point.prefill_tps), [1000, 500]);
  assert.equal(points[1].decode_kind, "long");
  assert.equal(points[1].prefill_context_tokens, 10256);
  assert.equal(points[1].decode_context_tokens, 11256);
});

test("cycle memory uses peak swap rather than a recovered end-of-phase value", () => {
  const phases = [
    { phase_id: 1, phase: "prefill", context_end_tokens: 5000, tokens_per_second: 1000, swap_growth_peak_bytes: 8 * 1024 ** 3, swap_used_peak_bytes: 10 * 1024 ** 3 },
    { phase_id: 2, phase: "decode", decode_kind: "short", context_end_tokens: 5256, tokens_per_second: 100, swap_used_end_bytes: 2 * 1024 ** 3 },
  ];
  const [point] = buildCyclePoints(phases, [], 2 * 1024 ** 3);
  assert.equal(point.swap_growth_gib, 8);
  assert.equal(point.swap_total_gib, 10);
});

test("model benchmarks group capacity and useful-task runs", () => {
  const modelBenchmarks = buildModelBenchmarks([
    {
      id: "capacity-one", kind: "capacity-run", model: "/models/Qwen",
      modelName: "Qwen", runAt: "2026-09-17T01:00:00.000Z", runAtEpoch: 1,
      maxContextTokens: 1000, finalDecodeTps: 40, peakMlxActiveGib: 2,
      peakSwapGrowthGib: 1, phaseCount: 4,
    },
    {
      id: "capacity-two", kind: "capacity-run", model: "/models/Qwen",
      modelName: "Qwen", runAt: "2026-09-17T02:00:00.000Z", runAtEpoch: 2,
      maxContextTokens: 2000, finalDecodeTps: 60, peakMlxActiveGib: 4,
      peakSwapGrowthGib: 3, phaseCount: 6,
    },
    {
      id: "tasks", kind: "useful-task-run", model: "Qwen", modelName: "Qwen",
      runAt: "2026-09-17T03:00:00.000Z", runAtEpoch: 3,
      maxContextTokens: 310, peakMlxActiveGib: 3, observationCount: 2,
    },
  ]);

  assert.equal(modelBenchmarks.length, 1);
  assert.equal(modelBenchmarks[0].runCount, 2);
  assert.equal(modelBenchmarks[0].usefulTaskRunCount, 1);
  assert.equal(modelBenchmarks[0].observationCount, 2);
  assert.equal(modelBenchmarks[0].maxContextTokens, 2000);
  assert.equal(modelBenchmarks[0].finalDecodeTps, 50);
  assert.equal(modelBenchmarks[0].peakMlxActiveGib, 3);
});

test("model benchmarks group runtime and quantization variants by base model", () => {
  const variants = [
    "Qwen3.5-0.8B",
    "Qwen3.5-0.8B-MLX-bf16",
    "Qwen3.5-0.8B-OptiQ-4bit",
    "Qwen3.5-0.8B · MTP",
  ];
  const modelBenchmarks = buildModelBenchmarks(variants.map((modelName, index) => ({
    id: `run-${index}`,
    kind: "capacity-run",
    model: `/models/${modelName}`,
    modelName,
    runAt: `2026-09-17T0${index + 1}:00:00.000Z`,
    runAtEpoch: index + 1,
    maxContextTokens: 1000,
    finalDecodeTps: 20 + (index * 10),
    peakMlxActiveGib: 2 + index,
    phaseCount: 1,
  })));

  assert.equal(modelBenchmarks.length, 1);
  assert.equal(modelBenchmarks[0].modelName, "Qwen3.5-0.8B");
  assert.equal(modelBenchmarks[0].runCount, 4);
  assert.equal(modelBenchmarks[0].finalDecodeTps, 35);
  assert.equal(modelBenchmarks[0].peakMlxActiveGib, 3.5);
});

test("canonical model names remove known packaging suffixes", () => {
  assert.equal(canonicalModelName("gemma-4-26B-A4B-it-QAT-MLX-4bit"), "gemma-4-26B-A4B-it");
  assert.equal(canonicalModelName("Qwen3.6-35B-A3B-4bit"), "Qwen3.6-35B-A3B");
  assert.equal(canonicalModelName("gpt-oss-20b-MXFP4-Q8"), "gpt-oss-20b");
});

test("model detail averages matching contexts into one line", () => {
  const modelBenchmark = {
    id: "model-Qwen", kind: "model", modelName: "Qwen",
    sourceIds: ["one", "two"], runCount: 2, usefulTaskRunCount: 1,
    observationCount: 2,
  };
  const aggregate = aggregateBenchmarkDetails(modelBenchmark, [
    { points: [
      { context_tokens: 1000, prefill_tps: 100, decode_tps: 20 },
      { point_kind: "useful-task", context_tokens: 300, task_prompt_tokens: 300, task_generation_tps: 40 },
    ] },
    { points: [
      { context_tokens: 1000, prefill_tps: 200, decode_tps: 40 },
      { point_kind: "useful-task", context_tokens: 300, task_prompt_tokens: 300, task_generation_tps: 60 },
    ] },
  ]);

  const prefillPoint = aggregate.points.find((point) => point.point_kind === "capacity-prefill");
  const decodePoint = aggregate.points.find((point) => point.point_kind === "capacity-decode");
  const usefulTaskPoint = aggregate.points.find((point) => point.point_kind === "useful-task");
  assert.equal(prefillPoint.prefill_tps, 150);
  assert.equal(decodePoint.decode_tps, 30);
  assert.equal(prefillPoint.sample_count, 2);
  assert.equal(usefulTaskPoint.task_generation_tps, 50);
  assert.equal(usefulTaskPoint.sample_count, 2);
});

test("model averages ignore missing measurements", () => {
  const [modelBenchmark] = buildModelBenchmarks([
    {
      id: "one", kind: "capacity-run", model: "Qwen", modelName: "Qwen",
      runAt: "2026-09-17T01:00:00.000Z", runAtEpoch: 1,
      maxContextTokens: 1000, finalDecodeTps: 40, peakMlxActiveGib: null,
      phaseCount: 1,
    },
    {
      id: "two", kind: "capacity-run", model: "Qwen", modelName: "Qwen",
      runAt: "2026-09-17T02:00:00.000Z", runAtEpoch: 2,
      maxContextTokens: 2000, finalDecodeTps: null, peakMlxActiveGib: 4,
      phaseCount: 1,
    },
  ]);

  assert.equal(modelBenchmark.finalDecodeTps, 40);
  assert.equal(modelBenchmark.peakMlxActiveGib, 4);
});

test("model detail separates matching prefills from short and sustained decodes", () => {
  const modelBenchmark = {
    id: "model-Qwen", kind: "model", modelName: "Qwen",
    sourceIds: ["one", "two"], runCount: 2, usefulTaskRunCount: 0,
    observationCount: 0,
  };
  const aggregate = aggregateBenchmarkDetails(modelBenchmark, [
    { points: [{
      cycle: 1, context_tokens: 1256, prefill_context_tokens: 1000,
      decode_context_tokens: 1256, prefill_tps: 100, decode_tps: 40,
      decode_kind: "short",
    }] },
    { points: [{
      cycle: 1, context_tokens: 2000, prefill_context_tokens: 1000,
      decode_context_tokens: 2000, prefill_tps: 200, decode_tps: 20,
      decode_kind: "long",
    }] },
  ]);

  const prefillPoints = aggregate.points.filter(
    (point) => point.point_kind === "capacity-prefill",
  );
  const decodePoints = aggregate.points.filter(
    (point) => point.point_kind === "capacity-decode",
  );
  assert.equal(prefillPoints.length, 1);
  assert.equal(prefillPoints[0].prefill_tps, 150);
  assert.equal(prefillPoints[0].sample_count, 2);
  assert.deepEqual(decodePoints.map((point) => point.decode_kind), ["short", "long"]);
  assert.deepEqual(decodePoints.map((point) => point.decode_tps), [40, 20]);
});
