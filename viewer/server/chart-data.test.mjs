import assert from "node:assert/strict";
import test from "node:test";

import { chartPoint, individualBenchmarksForModel } from "../src/chart-data.js";

test("individual benchmarks use aggregate source IDs across model variants", () => {
  const modelBenchmark = {
    modelName: "Qwen3.6-35B-A3B",
    sourceIds: ["raw-run", "speculative-run"],
  };
  const benchmarks = [
    { id: "raw-run", modelName: "Qwen3.6-35B-A3B-4bit" },
    { id: "speculative-run", modelName: "Qwen3.6-35B-A3B-4bit · MTP" },
    { id: "other-run", modelName: "Qwen3.5-0.8B" },
  ];

  assert.deepEqual(
    individualBenchmarksForModel(modelBenchmark, benchmarks).map((benchmark) => benchmark.id),
    ["raw-run", "speculative-run"],
  );
});

test("chart points omit missing measurements instead of coercing them to zero", () => {
  const memoryPoint = {
    point_kind: "capacity-memory",
    context_tokens: 4096,
    decode_context_tokens: 4096,
    prefill_tps: null,
  };
  assert.equal(chartPoint(memoryPoint, "prefill_tps"), null);
});

test("chart points retain genuine zero measurements", () => {
  assert.deepEqual(
    chartPoint({ context_tokens: 4096, decode_context_tokens: 4096, swap_growth_gib: 0 }, "swap_growth_gib"),
    { x: 4096, y: 0, cycle: undefined, decodeKind: undefined },
  );
});
