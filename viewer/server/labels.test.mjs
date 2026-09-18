import assert from "node:assert/strict";
import test from "node:test";

import { legendLabelMode, seriesLabel } from "../src/labels.js";

const runs = {
  first: { modelName: "Model A", runAt: "2026-09-17T01:00:00" },
  second: { modelName: "Model B", runAt: "2026-09-17T02:00:00" },
  repeat: { modelName: "Model A", runAt: "2026-09-17T03:00:00" },
  modelA: { kind: "model", modelName: "Model A", runAt: "2026-09-17T04:00:00" },
  modelB: { kind: "model", modelName: "Model B", runAt: "2026-09-17T05:00:00" },
};
const speed = { label: "Prefill speed" };

test("a single run uses metric-only legend labels", () => {
  const series = [
    { benchmarkId: "first", metric: "prefill_tps" },
    { benchmarkId: "first", metric: "swap_growth_gib" },
  ];
  const mode = legendLabelMode(series, runs);
  assert.equal(mode, "metric");
  assert.equal(seriesLabel(runs.first, speed, mode), "Prefill speed");
});

test("distinct models omit run times", () => {
  const mode = legendLabelMode(
    [
      { benchmarkId: "first", metric: "prefill_tps" },
      { benchmarkId: "second", metric: "prefill_tps" },
    ],
    runs,
  );
  assert.equal(mode, "model");
  assert.equal(seriesLabel(runs.first, speed, mode), "Model A");
});

test("distinct models retain metric names when metrics differ", () => {
  const mode = legendLabelMode(
    [
      { benchmarkId: "first", metric: "prefill_tps" },
      { benchmarkId: "second", metric: "decode_tps" },
    ],
    runs,
  );
  assert.equal(mode, "model-metric");
  assert.equal(seriesLabel(runs.first, speed, mode), "Model A · Prefill speed");
});

test("repeated models include run times", () => {
  const mode = legendLabelMode(
    [{ benchmarkId: "first" }, { benchmarkId: "repeat" }],
    runs,
  );
  assert.equal(mode, "run");
  assert.match(seriesLabel(runs.first, speed, mode), /^Model A · .+ · Prefill speed$/);
});

test("aggregate models never include evidence timestamps", () => {
  const mode = legendLabelMode(
    [
      { benchmarkId: "modelA", metric: "prefill_tps" },
      { benchmarkId: "modelB", metric: "prefill_tps" },
    ],
    runs,
  );
  assert.equal(mode, "model");
  assert.equal(seriesLabel(runs.modelA, speed, mode), "Model A");
});
