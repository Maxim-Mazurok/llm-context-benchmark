const timestampPattern = /^(\d+)\.(\d{2})\.(\d{3})\.(\d{3})\s/;

function elapsedSeconds(line) {
  const match = line.match(timestampPattern);
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]) + Number(match[3]) / 1_000 + Number(match[4]) / 1_000_000;
}

function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * ratio))];
}

export function parseLlamaServerLog(content) {
  const tasks = new Map();
  const records = [];
  let pendingSelection = null;
  let contextSize = null;
  let model = null;

  for (const [lineIndex, line] of content.split(/\r?\n/).entries()) {
    const elapsed = elapsedSeconds(line);
    if (elapsed == null) continue;

    const loadMatch = line.match(/load_model: loading model '([^']+)'/);
    if (loadMatch) model = loadMatch[1];

    const contextMatch = line.match(/n_ctx_slot = (\d+)/);
    if (contextMatch) contextSize = Number(contextMatch[1]);

    const selectionMatch = line.match(/selected slot by LCP similarity, f_sim_best = ([\d.]+).*f_keep = ([\d.]+)/);
    if (selectionMatch) {
      pendingSelection = {
        lcpSimilarity: Number(selectionMatch[1]),
        keepFraction: Number(selectionMatch[2]),
      };
      continue;
    }

    const launchMatch = line.match(/task (\d+) \| processing task/);
    if (launchMatch) {
      const taskId = Number(launchMatch[1]);
      tasks.set(taskId, {
        taskId,
        line: lineIndex + 1,
        startedSeconds: elapsed,
        lcpSimilarity: pendingSelection?.lcpSimilarity ?? null,
        keepFraction: pendingSelection?.keepFraction ?? null,
      });
      pendingSelection = null;
      continue;
    }

    const timingMatch = line.match(/task (\d+) \|\s+(prompt eval time|eval time|total time) =\s+([\d.]+) ms \/\s+(\d+) tokens/);
    if (timingMatch) {
      const task = tasks.get(Number(timingMatch[1]));
      if (!task) continue;
      const durationSeconds = Number(timingMatch[3]) / 1_000;
      const tokens = Number(timingMatch[4]);
      if (timingMatch[2] === "prompt eval time") {
        task.prefillDurationSeconds = durationSeconds;
        task.prefillTokens = tokens;
        task.prefillTokensPerSecond = tokens / durationSeconds;
      } else if (timingMatch[2] === "eval time") {
        task.decodeDurationSeconds = durationSeconds;
        task.decodeTokens = tokens;
        task.decodeTokensPerSecond = tokens / durationSeconds;
      } else {
        task.totalDurationSeconds = durationSeconds;
      }
      continue;
    }

    const releaseMatch = line.match(/task (\d+) \| stop processing: n_tokens = (\d+), truncated = (\d+)/);
    if (!releaseMatch) continue;
    const taskId = Number(releaseMatch[1]);
    const task = tasks.get(taskId);
    if (!task) continue;
    task.finishedSeconds = elapsed;
    task.contextTokens = Number(releaseMatch[2]);
    task.truncated = releaseMatch[3] === "1";
    task.promptTokens = Math.max(0, task.contextTokens - (task.decodeTokens || 0) + 1);
    task.cachedPromptTokens = Math.max(0, task.promptTokens - (task.prefillTokens || 0));
    task.effectivePrefillTokensPerSecond = task.prefillDurationSeconds > 0
      ? task.promptTokens / task.prefillDurationSeconds
      : null;
    task.estimatedNoCachePrefillDurationSeconds = task.prefillTokensPerSecond > 0
      ? task.promptTokens / task.prefillTokensPerSecond
      : null;
    task.estimatedCacheSavedSeconds = task.estimatedNoCachePrefillDurationSeconds == null
      ? null
      : Math.max(0, task.estimatedNoCachePrefillDurationSeconds - task.prefillDurationSeconds);
    task.contextBeforeDecodeTokens = task.promptTokens;
    records.push(task);
    tasks.delete(taskId);
  }

  const completeRecords = records.filter((record) => record.prefillTokensPerSecond || record.decodeTokensPerSecond);
  const decodeRates = completeRecords.map((record) => record.decodeTokensPerSecond).filter(Number.isFinite);
  const prefillRates = completeRecords.map((record) => record.prefillTokensPerSecond).filter(Number.isFinite);
  const lastFinishedSeconds = Math.max(0, ...completeRecords.map((record) => record.finishedSeconds || 0));
  const totalPrefillSeconds = completeRecords.reduce((total, record) => total + (record.prefillDurationSeconds || 0), 0);
  const totalDecodeSeconds = completeRecords.reduce((total, record) => total + (record.decodeDurationSeconds || 0), 0);
  const estimatedNoCachePrefillSeconds = completeRecords.reduce(
    (total, record) => total + (record.estimatedNoCachePrefillDurationSeconds || 0),
    0,
  );

  return {
    metadata: {
      model,
      contextSize,
      durationSeconds: lastFinishedSeconds,
      parsedLines: content.split(/\r?\n/).length,
      completedTasks: completeRecords.length,
    },
    summary: {
      peakContextTokens: Math.max(0, ...completeRecords.map((record) => record.contextTokens || 0)),
      medianPrefillTokensPerSecond: percentile(prefillRates, 0.5),
      medianDecodeTokensPerSecond: percentile(decodeRates, 0.5),
      p10DecodeTokensPerSecond: percentile(decodeRates, 0.1),
      totalPrefillSeconds,
      totalDecodeSeconds,
      estimatedNoCachePrefillSeconds,
      estimatedCacheSavedSeconds: Math.max(0, estimatedNoCachePrefillSeconds - totalPrefillSeconds),
      cacheHitTasks: completeRecords.filter((record) => record.lcpSimilarity != null).length,
    },
    records: completeRecords,
  };
}