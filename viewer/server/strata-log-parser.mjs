function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * ratio;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction;
}

function finiteNumber(value) {
  if (value == null || value === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function parseStrataMetrics(content) {
  const records = [];
  let rejectedLines = 0;

  for (const [lineIndex, line] of content.split(/\r?\n/).entries()) {
    if (!line.trim()) continue;
    let raw;
    try {
      raw = JSON.parse(line);
    } catch {
      rejectedLines += 1;
      continue;
    }

    const time = finiteNumber(raw.time);
    const promptTokens = finiteNumber(raw.prompt_tokens);
    const reusedTokens = finiteNumber(raw.reused) ?? 0;
    const outputTokens = finiteNumber(raw.output_tokens) ?? 0;
    const prefillTokensPerSecond = finiteNumber(raw.prefill_tok_s);
    const decodeTokensPerSecond = finiteNumber(raw.decode_tok_s);
    if (time == null || promptTokens == null || (prefillTokensPerSecond == null && decodeTokensPerSecond == null)) {
      rejectedLines += 1;
      continue;
    }

    records.push({
      request: records.length + 1,
      line: lineIndex + 1,
      time,
      durationSeconds: finiteNumber(raw.duration_s),
      finishReason: raw.finish || null,
      promptTokens,
      reusedTokens,
      freshPromptTokens: Math.max(0, promptTokens - reusedTokens),
      outputTokens,
      prefillDurationSeconds: (finiteNumber(raw.prompt_ms) ?? 0) / 1_000,
      decodeDurationSeconds: (finiteNumber(raw.decode_ms) ?? 0) / 1_000,
      prefillTokensPerSecond,
      decodeTokensPerSecond,
      contextTokens: promptTokens,
    });
  }

  records.sort((left, right) => left.time - right.time);
  const startedAt = records[0]?.time ?? null;
  for (const [index, record] of records.entries()) {
    record.request = index + 1;
    record.elapsedSeconds = startedAt == null ? 0 : record.time - startedAt;
    record.cacheReusePercent = record.promptTokens > 0 ? record.reusedTokens / record.promptTokens * 100 : 0;
  }

  const prefillRates = records.map((record) => record.prefillTokensPerSecond).filter(Number.isFinite);
  const decodeRates = records.map((record) => record.decodeTokensPerSecond).filter(Number.isFinite);
  const lastRecord = records.at(-1);
  return {
    metadata: {
      startedAt: startedAt == null ? null : new Date(startedAt * 1_000).toISOString(),
      endedAt: lastRecord ? new Date(lastRecord.time * 1_000).toISOString() : null,
      durationSeconds: startedAt == null || !lastRecord ? 0 : lastRecord.time - startedAt,
      parsedLines: content.split(/\r?\n/).filter((line) => line.trim()).length,
      rejectedLines,
      completedRequests: records.length,
    },
    summary: {
      peakContextTokens: Math.max(0, ...records.map((record) => record.contextTokens)),
      medianPrefillTokensPerSecond: percentile(prefillRates, 0.5),
      medianDecodeTokensPerSecond: percentile(decodeRates, 0.5),
      p10DecodeTokensPerSecond: percentile(decodeRates, 0.1),
      p90DecodeTokensPerSecond: percentile(decodeRates, 0.9),
      totalPrefillSeconds: records.reduce((total, record) => total + record.prefillDurationSeconds, 0),
      totalDecodeSeconds: records.reduce((total, record) => total + record.decodeDurationSeconds, 0),
      reusedPromptTokens: records.reduce((total, record) => total + record.reusedTokens, 0),
      freshPromptTokens: records.reduce((total, record) => total + record.freshPromptTokens, 0),
      cacheHitRequests: records.filter((record) => record.reusedTokens > 0).length,
    },
    records,
  };
}

export function summarizeStrataBins(records, binSize = 10_000) {
  const bins = new Map();
  for (const record of records) {
    const floor = Math.floor(record.contextTokens / binSize) * binSize;
    const bin = bins.get(floor) || { floor, prefill: [], decode: [], count: 0 };
    if (Number.isFinite(record.prefillTokensPerSecond) && record.freshPromptTokens >= 128) {
      bin.prefill.push(record.prefillTokensPerSecond);
    }
    if (Number.isFinite(record.decodeTokensPerSecond) && record.outputTokens >= 32) {
      bin.decode.push(record.decodeTokensPerSecond);
    }
    bin.count += 1;
    bins.set(floor, bin);
  }

  return [...bins.values()].sort((left, right) => left.floor - right.floor).map((bin) => ({
    contextTokens: bin.floor + binSize / 2,
    requestCount: bin.count,
    prefill: bin.prefill.length ? {
      count: bin.prefill.length,
      q1: percentile(bin.prefill, 0.25),
      median: percentile(bin.prefill, 0.5),
      q3: percentile(bin.prefill, 0.75),
    } : null,
    decode: bin.decode.length ? {
      count: bin.decode.length,
      q1: percentile(bin.decode, 0.25),
      median: percentile(bin.decode, 0.5),
      q3: percentile(bin.decode, 0.75),
    } : null,
  }));
}