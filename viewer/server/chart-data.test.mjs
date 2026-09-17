import assert from "node:assert/strict";
import test from "node:test";

import { chartPoint } from "../src/chart-data.js";

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
