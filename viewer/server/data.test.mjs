import assert from "node:assert/strict";
import test from "node:test";

import { buildCyclePoints } from "./data.mjs";

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
