import assert from "node:assert/strict";
import test from "node:test";

import { parseStrataMetrics, summarizeStrataBins } from "./strata-log-parser.mjs";

const fixture = [
  { time: 1000, duration_s: 5, finish: "stop", prompt_tokens: 1000, reused: 0, output_tokens: 64, prompt_ms: 20000, decode_ms: 4000, prefill_tok_s: 50, decode_tok_s: 16 },
  { time: 1060, duration_s: 8, finish: "length", prompt_tokens: 12000, reused: 11000, output_tokens: 40, prompt_ms: 25000, decode_ms: 3000, prefill_tok_s: 40, decode_tok_s: 13.333 },
  { time: 1120, duration_s: 2, finish: "stop", prompt_tokens: 12500, reused: 12450, output_tokens: 1, prompt_ms: 2000, decode_ms: 70, prefill_tok_s: 25, decode_tok_s: 14.286 },
];

test("parses durable Strata metrics and derives cache-aware fields", () => {
  const result = parseStrataMetrics(`${fixture.map(JSON.stringify).join("\n")}\nnot-json\n`);

  assert.equal(result.metadata.completedRequests, 3);
  assert.equal(result.metadata.rejectedLines, 1);
  assert.equal(result.metadata.durationSeconds, 120);
  assert.equal(result.summary.peakContextTokens, 12500);
  assert.equal(result.summary.cacheHitRequests, 2);
  assert.equal(result.summary.reusedPromptTokens, 23450);
  assert.equal(result.records[1].freshPromptTokens, 1000);
  assert.equal(result.records[1].cacheReusePercent, 11000 / 12000 * 100);
  assert.equal(result.records[2].elapsedSeconds, 120);
});

test("builds context bins with median and interquartile throughput", () => {
  const records = parseStrataMetrics(fixture.map(JSON.stringify).join("\n")).records;
  const bins = summarizeStrataBins(records, 10_000);

  assert.equal(bins.length, 2);
  assert.equal(bins[0].prefill.median, 50);
  assert.equal(bins[0].decode.median, 16);
  assert.equal(bins[1].prefill.median, 40);
  assert.equal(bins[1].decode.median, 13.333);
  assert.equal(bins[1].decode.count, 1);
});

test("rejects records whose required numeric fields are null", () => {
  const content = [
    { time: null, prompt_tokens: 100, prefill_tok_s: 10 },
    { time: 1000, prompt_tokens: null, prefill_tok_s: 10 },
    { time: 1000, prompt_tokens: 100, prefill_tok_s: null, decode_tok_s: null },
  ].map(JSON.stringify).join("\n");

  const result = parseStrataMetrics(content);

  assert.equal(result.metadata.completedRequests, 0);
  assert.equal(result.metadata.rejectedLines, 3);
});