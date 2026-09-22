import assert from "node:assert/strict";
import test from "node:test";

import { parseLlamaServerLog } from "./llama-log-parser.mjs";

test("joins llama slot lifecycle into performance records", () => {
  const result = parseLlamaServerLog(`0.00.142.891 I srv load_model: loading model '/models/Qwen.gguf'
0.32.374.642 I srv load_model: initializing, n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'
18.19.708.824 I slot get_availabl: id 0 | task -1 | selected slot by LCP similarity, f_sim_best = 0.995 (> 0.100 thold), f_keep = 1.000
18.19.709.100 I slot launch_slot_: id 0 | task 4628 | processing task, is_child = 0
20.12.657.785 I slot print_timing: id 0 | task 4628 | prompt eval time = 3156.63 ms / 242 tokens (13.04 ms per token, 76.66 tokens per second)
20.12.657.800 I slot print_timing: id 0 | task 4628 | eval time = 109650.00 ms / 500 tokens (219.30 ms per token, 4.56 tokens per second)
20.12.657.810 I slot print_timing: id 0 | task 4628 | total time = 112806.63 ms / 742 tokens
20.12.660.000 I slot release: id 0 | task 4628 | stop processing: n_tokens = 20000, truncated = 0`);

  assert.equal(result.metadata.model, "/models/Qwen.gguf");
  assert.equal(result.metadata.contextSize, 262144);
  assert.equal(result.metadata.durationSeconds, 20 * 60 + 12.66);
  assert.equal(result.summary.peakContextTokens, 20000);
  assert.equal(result.summary.cacheHitTasks, 1);
  assert.equal(result.records.length, 1);
  assert.deepEqual(result.records[0], {
    taskId: 4628,
    line: 4,
    startedSeconds: 18 * 60 + 19.7091,
    lcpSimilarity: 0.995,
    keepFraction: 1,
    prefillDurationSeconds: result.records[0].prefillDurationSeconds,
    prefillTokens: 242,
    prefillTokensPerSecond: result.records[0].prefillTokensPerSecond,
    decodeDurationSeconds: 109.65,
    decodeTokens: 500,
    decodeTokensPerSecond: 500 / 109.65,
    totalDurationSeconds: 112.80663,
    finishedSeconds: 20 * 60 + 12.66,
    contextTokens: 20000,
    truncated: false,
    promptTokens: 19501,
    cachedPromptTokens: 19259,
    effectivePrefillTokensPerSecond: result.records[0].effectivePrefillTokensPerSecond,
    estimatedNoCachePrefillDurationSeconds: result.records[0].estimatedNoCachePrefillDurationSeconds,
    estimatedCacheSavedSeconds: result.records[0].estimatedCacheSavedSeconds,
    contextBeforeDecodeTokens: 19501,
  });
  assert.ok(Math.abs(result.records[0].prefillDurationSeconds - 3.15663) < 1e-9);
  assert.ok(Math.abs(result.records[0].prefillTokensPerSecond - (242 / 3.15663)) < 1e-9);
  assert.ok(Math.abs(result.records[0].effectivePrefillTokensPerSecond - (19501 / 3.15663)) < 1e-9);
  assert.ok(Math.abs(result.records[0].estimatedNoCachePrefillDurationSeconds - (19501 / (242 / 3.15663))) < 1e-9);
});

test("counts cached prompt tokens in effective prefill throughput", () => {
  const result = parseLlamaServerLog(`1.00.000.000 I slot launch_slot_: id 0 | task 8 | processing task, is_child = 0
1.05.000.000 I slot print_timing: id 0 | task 8 | prompt eval time = 5000.00 ms / 100 tokens
1.05.100.000 I slot print_timing: id 0 | task 8 | eval time = 100.00 ms / 1 tokens
1.05.101.000 I slot release: id 0 | task 8 | stop processing: n_tokens = 15100, truncated = 0`);

  assert.equal(result.records[0].promptTokens, 15100);
  assert.equal(result.records[0].cachedPromptTokens, 15000);
  assert.equal(result.records[0].effectivePrefillTokensPerSecond, 3020);
  assert.equal(result.records[0].estimatedNoCachePrefillDurationSeconds, 755);
  assert.equal(result.records[0].estimatedCacheSavedSeconds, 750);
  assert.equal(result.summary.estimatedNoCachePrefillSeconds, 755);
  assert.equal(result.summary.estimatedCacheSavedSeconds, 750);
});

test("ignores unfinished tasks and tolerates unrelated lines", () => {
  const result = parseLlamaServerLog(`not a llama line
1.00.000.000 I slot launch_slot_: id 0 | task 7 | processing task, is_child = 0
1.01.000.000 I slot print_timing: id 0 | task 7 | prompt eval time = 1000.00 ms / 100 tokens
1.02.000.000 I slot release: id 0 | task 9 | stop processing: n_tokens = 100, truncated = 0`);
  assert.equal(result.records.length, 0);
});