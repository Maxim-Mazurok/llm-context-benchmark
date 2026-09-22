import { useEffect, useMemo, useState } from "react";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

import { fetchLlamaLog } from "./api.js";

ChartJS.register(BarElement, CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

const COLORS = {
  prefill: "#b74228",
  decode: "#254b76",
  context: "#2f766f",
  suffix: "#9b6b00",
  cache: "#8f3f65",
  total: "#684f8e",
};

function formatNumber(value, digits = 0) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${formatNumber(seconds, 1)}s`;
  return `${Math.floor(seconds / 3_600)}h ${Math.floor((seconds % 3_600) / 60)}m`;
}

function point(record, horizontalValue, verticalValue) {
  if (!Number.isFinite(horizontalValue) || !Number.isFinite(verticalValue)) return null;
  return {
    x: horizontalValue,
    y: verticalValue,
    taskId: record.taskId,
    contextTokens: record.contextTokens,
    elapsedSeconds: record.finishedSeconds,
    promptTokens: record.promptTokens,
    cachedPromptTokens: record.cachedPromptTokens,
    prefillTokens: record.prefillTokens,
  };
}

function dataset(label, color, data, options = {}) {
  return {
    label,
    data: data.filter(Boolean),
    borderColor: color,
    backgroundColor: color,
    borderWidth: 1.75,
    pointRadius: 2.5,
    pointHoverRadius: 6,
    showLine: options.showLine ?? true,
    tension: options.tension ?? 0.08,
    yAxisID: options.yAxisID || "y",
  };
}

function PerformanceChart({ title, subtitle, datasets, horizontalLabel, axes = {} }) {
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 260, easing: "easeOutQuart" },
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { color: "#443b31", usePointStyle: true, boxWidth: 8 } },
      tooltip: {
        callbacks: {
          title(items) {
            const raw = items[0]?.raw;
            return `Task ${raw?.taskId} · ${formatNumber(raw?.contextTokens)} context tokens`;
          },
          afterLabel(context) {
            const raw = context.raw;
            const details = [`Elapsed ${formatDuration(raw?.elapsedSeconds)}`];
            if (context.dataset.cacheAdjusted) {
              details.push(`Full prompt ${formatNumber(raw?.promptTokens)} tokens`);
              details.push(`Cached ${formatNumber(raw?.cachedPromptTokens)} · evaluated ${formatNumber(raw?.prefillTokens)}`);
            }
            return details;
          },
        },
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: horizontalLabel },
        grid: { color: "#ddd4c5" },
        ticks: { color: "#6f675c", maxTicksLimit: 8 },
      },
      y: {
        type: "linear",
        beginAtZero: true,
        title: { display: true, text: axes.y || "Tokens per second" },
        grid: { color: "#ddd4c5" },
        ticks: { color: "#6f675c", maxTicksLimit: 7 },
      },
      ...(axes.ySecondary ? {
        ySecondary: {
          type: "linear",
          position: "right",
          beginAtZero: true,
          max: axes.ySecondaryMax,
          title: { display: true, text: axes.ySecondary },
          grid: { color: "transparent" },
          ticks: { color: "#6f675c", maxTicksLimit: 7 },
        },
      } : {}),
    },
  }), [axes, horizontalLabel]);

  return (
    <section className="log-chart">
      <div className="log-chart-heading"><h2>{title}</h2><p>{subtitle}</p></div>
      <div className="log-chart-frame"><Line data={{ datasets }} options={options} /></div>
    </section>
  );
}

function TimeBreakdownChart({ title, subtitle, breakdown, percentage = false }) {
  const options = useMemo(() => ({
    indexAxis: "y",
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 260, easing: "easeOutQuart" },
    plugins: {
      legend: { labels: { color: "#443b31", usePointStyle: true, boxWidth: 8 } },
      tooltip: {
        callbacks: {
          label(context) {
            const value = context.raw;
            return `${context.dataset.label}: ${percentage ? `${formatNumber(value, 1)}%` : formatDuration(value)}`;
          },
          footer(items) {
            const total = items[0]?.dataset?.totals?.[items[0].dataIndex];
            return percentage ? "" : `Total: ${formatDuration(total)}`;
          },
        },
      },
    },
    scales: {
      x: {
        stacked: true,
        beginAtZero: true,
        max: percentage ? 100 : undefined,
        title: { display: true, text: percentage ? "Share of prompt evaluation and decode time (%)" : "Compute time" },
        grid: { color: "#ddd4c5" },
        ticks: {
          color: "#6f675c",
          callback(value) {
            return percentage ? `${value}%` : formatDuration(value);
          },
        },
      },
      y: {
        stacked: true,
        grid: { display: false },
        ticks: { color: "#443b31" },
      },
    },
  }), [percentage]);

  return (
    <section className="log-chart">
      <div className="log-chart-heading"><h2>{title}</h2><p>{subtitle}</p></div>
      <div className="log-chart-frame"><Bar data={breakdown} options={options} /></div>
    </section>
  );
}

function buildTimeBreakdown(records) {
  const actualPrefillSeconds = records.reduce((total, record) => total + (record.prefillDurationSeconds || 0), 0);
  const decodeSeconds = records.reduce((total, record) => total + (record.decodeDurationSeconds || 0), 0);
  const estimatedNoCachePrefillSeconds = records.reduce(
    (total, record) => total + (record.estimatedNoCachePrefillDurationSeconds || 0),
    0,
  );
  const actualTotalSeconds = actualPrefillSeconds + decodeSeconds;
  const estimatedNoCacheTotalSeconds = estimatedNoCachePrefillSeconds + decodeSeconds;
  const totals = [actualTotalSeconds, estimatedNoCacheTotalSeconds];
  const labels = ["Measured with prefix reuse", "Estimated without prefix reuse"];
  const absolute = {
    labels,
    datasets: [
      { ...dataset("Prompt evaluation", COLORS.prefill, [actualPrefillSeconds, estimatedNoCachePrefillSeconds]), totals },
      { ...dataset("Decode", COLORS.decode, [decodeSeconds, decodeSeconds]), totals },
    ],
  };
  const percentage = {
    labels,
    datasets: absolute.datasets.map((series) => ({
      ...series,
      data: series.data.map((value, index) => totals[index] > 0 ? value / totals[index] * 100 : 0),
    })),
  };

  return {
    absolute,
    percentage,
    actualTotalSeconds,
    estimatedNoCacheTotalSeconds,
    estimatedCacheSavedSeconds: Math.max(0, estimatedNoCacheTotalSeconds - actualTotalSeconds),
    overallSpeedup: actualTotalSeconds > 0 ? estimatedNoCacheTotalSeconds / actualTotalSeconds : null,
  };
}

function buildCharts(records) {
  const elapsedHours = (record) => record.finishedSeconds / 3_600;
  const contextThousands = (record) => record.contextBeforeDecodeTokens / 1_000;
  const contextFraction = (record) => record.contextTokens > 0
    ? record.prefillTokens / record.contextTokens * 100
    : null;
  return {
    speedOverTime: [
      dataset("Prefill", COLORS.prefill, records.map((record) => point(record, elapsedHours(record), record.prefillTokensPerSecond))),
      dataset("Decode", COLORS.decode, records.map((record) => point(record, elapsedHours(record), record.decodeTokensPerSecond))),
    ],
    speedByContext: [
      dataset("Prefill", COLORS.prefill, records.map((record) => point(record, contextThousands(record), record.prefillTokensPerSecond)), { showLine: false }),
      dataset("Decode", COLORS.decode, records.map((record) => point(record, contextThousands(record), record.decodeTokensPerSecond)), { showLine: false }),
    ],
    effectiveSpeedOverTime: [
      {
        ...dataset("Cache-adjusted prefill", COLORS.cache, records.map((record) => point(record, elapsedHours(record), record.effectivePrefillTokensPerSecond))),
        cacheAdjusted: true,
      },
    ],
    effectiveSpeedByContext: [
      {
        ...dataset("Cache-adjusted prefill", COLORS.cache, records.map((record) => point(record, contextThousands(record), record.effectivePrefillTokensPerSecond)), { showLine: false }),
        cacheAdjusted: true,
      },
    ],
    contextOverTime: [
      dataset("Retained context", COLORS.context, records.map((record) => point(record, elapsedHours(record), record.contextTokens / 1_000))),
      dataset("Evaluated suffix", COLORS.suffix, records.map((record) => point(record, elapsedHours(record), record.prefillTokens / 1_000))),
    ],
    latencyByContext: [
      dataset("Prefill", COLORS.prefill, records.map((record) => point(record, contextThousands(record), record.prefillDurationSeconds)), { showLine: false }),
      dataset("Decode", COLORS.decode, records.map((record) => point(record, contextThousands(record), record.decodeDurationSeconds)), { showLine: false }),
      dataset("Total request", COLORS.total, records.map((record) => point(record, contextThousands(record), record.totalDurationSeconds)), { showLine: false }),
    ],
    cacheReuse: [
      dataset("New prompt share", COLORS.suffix, records.map((record) => point(record, contextThousands(record), contextFraction(record))), { showLine: false }),
      dataset("Prefix similarity", COLORS.cache, records.map((record) => point(record, contextThousands(record), record.lcpSimilarity == null ? null : record.lcpSimilarity * 100)), { showLine: false, yAxisID: "ySecondary" }),
    ],
    perTokenCost: [
      dataset("Prefill cost", COLORS.prefill, records.map((record) => point(record, contextThousands(record), record.prefillTokens ? record.prefillDurationSeconds * 1_000 / record.prefillTokens : null)), { showLine: false }),
      dataset("Decode cost", COLORS.decode, records.map((record) => point(record, contextThousands(record), record.decodeTokens ? record.decodeDurationSeconds * 1_000 / record.decodeTokens : null)), { showLine: false }),
    ],
  };
}

export function LlamaLogDashboard() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");
  const [windowHours, setWindowHours] = useState("all");

  useEffect(() => {
    fetchLlamaLog().then(setPayload).catch((requestError) => setError(requestError.message));
  }, []);

  const records = useMemo(() => {
    if (!payload) return [];
    if (windowHours === "all") return payload.records;
    const cutoff = payload.metadata.durationSeconds - Number(windowHours) * 3_600;
    return payload.records.filter((record) => record.finishedSeconds >= cutoff);
  }, [payload, windowHours]);
  const charts = useMemo(() => buildCharts(records), [records]);
  const timeBreakdown = useMemo(() => buildTimeBreakdown(records), [records]);

  if (error) return <main className="log-dashboard"><a href="#">← Context Atlas</a><h1>Log could not be parsed</h1><p>{error}</p></main>;
  if (!payload) return <main className="log-dashboard loading-state"><div className="eyebrow">Parsing captured process</div><h1>Reading llama log…</h1></main>;

  const { metadata, source, summary } = payload;
  return (
    <main id="main-content" className="log-dashboard">
      <nav className="log-nav"><a href="#">← Context Atlas</a><span>{source.name} · {formatNumber(source.bytes / 1_000_000, 1)} MB</span></nav>
      <header className="log-header">
        <div><div className="eyebrow">llama.cpp process record</div><h1>Inference under load</h1></div>
        <p>{metadata.completedTasks} requests reveal how prefix reuse, context length, and sustained operation reshape throughput.</p>
      </header>

      <dl className="log-facts">
        <div><dt>Observed runtime</dt><dd>{formatDuration(metadata.durationSeconds)}</dd></div>
        <div><dt>Peak context</dt><dd>{formatNumber(summary.peakContextTokens)} <small>tokens</small></dd></div>
        <div><dt>Median prefill</dt><dd>{formatNumber(summary.medianPrefillTokensPerSecond, 1)} <small>tok/s</small></dd></div>
        <div><dt>Median decode</dt><dd>{formatNumber(summary.medianDecodeTokensPerSecond, 2)} <small>tok/s</small></dd></div>
        <div><dt>Cache-matched tasks</dt><dd>{formatNumber(summary.cacheHitTasks)} <small>of {metadata.completedTasks}</small></dd></div>
        <div><dt>Measured compute</dt><dd>{formatDuration(summary.totalPrefillSeconds + summary.totalDecodeSeconds)}</dd></div>
      </dl>

      <section className="log-controls" aria-label="Log time range">
        <div><strong>Analysis window</strong><span>{records.length} completed requests shown</span></div>
        <div className="log-range" role="group" aria-label="Analysis window">
          {[{ value: "all", label: "Full run" }, { value: "3", label: "Last 3h" }, { value: "1", label: "Last hour" }].map((option) => (
            <button type="button" key={option.value} aria-pressed={windowHours === option.value} onClick={() => setWindowHours(option.value)}>{option.label}</button>
          ))}
        </div>
      </section>

      <aside className="log-interpretation-note">
        <strong>Two caches, two different effects.</strong>
        <p>The modeled savings above come from active-slot KV prefix reuse: llama.cpp retained prior conversation state and evaluated only the new suffix. They do not measure any benefit from <code>--cache-ram</code>.</p>
        <p><code>--cache-ram 8192</code> configured a separate 8 GiB prompt-state store. It pushed this run into heavy swap, so its net performance effect remains unknown and may have been negative. A future run will use a smaller non-zero value that leaves RAM headroom for model weights, then compare wall time, prefill, decode, and swap.</p>
      </aside>

      <div className="log-chart-grid">
        <TimeBreakdownChart title="Where compute time goes" subtitle="Prompt evaluation versus decode, comparing measured prefix reuse with a linear no-reuse estimate." breakdown={timeBreakdown.percentage} percentage />
        <TimeBreakdownChart title="Prefix-reuse impact in absolute time" subtitle={`${formatDuration(timeBreakdown.estimatedCacheSavedSeconds)} estimated prompt work avoided · ${formatNumber(timeBreakdown.overallSpeedup, 1)}× modeled compute ratio. No-reuse prefill extrapolates each task's observed prefill rate across its full prompt.`} breakdown={timeBreakdown.absolute} />
        <PerformanceChart title="Speed through the night" subtitle="Wall-clock drift exposes warmup, pressure, and workload changes." datasets={charts.speedOverTime} horizontalLabel="Elapsed hours" />
        <PerformanceChart title="Speed against context" subtitle="Scatter separates context scaling from when each request ran." datasets={charts.speedByContext} horizontalLabel="Context before decode (thousand tokens)" />
        <PerformanceChart title="Cache-adjusted speed through the night" subtitle="Full prompt tokens divided by measured prompt-evaluation time, including reused prefix tokens." datasets={charts.effectiveSpeedOverTime} horizontalLabel="Elapsed hours" axes={{ y: "Effective prompt tokens per second" }} />
        <PerformanceChart title="Cache-adjusted speed against context" subtitle="Effective throughput credits cached prefix tokens that avoided recomputation." datasets={charts.effectiveSpeedByContext} horizontalLabel="Full prompt context (thousand tokens)" axes={{ y: "Effective prompt tokens per second" }} />
        <PerformanceChart title="Context trajectory" subtitle="Retained KV context beside newly evaluated prompt suffix." datasets={charts.contextOverTime} horizontalLabel="Elapsed hours" axes={{ y: "Thousand tokens" }} />
        <PerformanceChart title="Request latency" subtitle="Time-to-first-token pressure beside generation duration and total latency." datasets={charts.latencyByContext} horizontalLabel="Context before decode (thousand tokens)" axes={{ y: "Seconds" }} />
        <PerformanceChart title="Prefix reuse" subtitle="Low new-prompt share with high prefix similarity means cache reuse succeeded." datasets={charts.cacheReuse} horizontalLabel="Context before decode (thousand tokens)" axes={{ y: "New prompt share (%)", ySecondary: "Prefix similarity (%)", ySecondaryMax: 100 }} />
        <PerformanceChart title="Cost per token" subtitle="Per-token latency makes long-context attention cost visible even on cache hits." datasets={charts.perTokenCost} horizontalLabel="Context before decode (thousand tokens)" axes={{ y: "Milliseconds per token" }} />
      </div>

      <footer className="log-provenance"><strong>{source.name}</strong><span>Modified {new Date(source.modifiedAt).toLocaleString()} · {formatNumber(metadata.parsedLines)} lines · context capacity {formatNumber(metadata.contextSize)}</span><code>{metadata.model}</code></footer>
    </main>
  );
}