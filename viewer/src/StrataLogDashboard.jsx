import { useEffect, useMemo, useState } from "react";
import {
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

import { fetchStrataLog } from "./api.js";

ChartJS.register(Filler, LinearScale, PointElement, LineElement, Tooltip, Legend);

const COLORS = {
  prefill: "#b74228",
  prefillBand: "rgba(183, 66, 40, 0.15)",
  decode: "#245e68",
  decodeBand: "rgba(36, 94, 104, 0.16)",
  context: "#806515",
  cache: "#8f3f65",
};

function formatNumber(value, digits = 0) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${formatNumber(seconds, 1)}s`;
  if (seconds < 3_600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(seconds / 3_600)}h ${Math.floor((seconds % 3_600) / 60)}m`;
}

function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * ratio;
  const lower = Math.floor(position);
  const fraction = position - lower;
  return sorted[lower] + (sorted[Math.min(lower + 1, sorted.length - 1)] - sorted[lower]) * fraction;
}

function contextBins(records, size = 10_000) {
  const groups = new Map();
  for (const record of records) {
    const floor = Math.floor(record.contextTokens / size) * size;
    const group = groups.get(floor) || { floor, prefill: [], decode: [], requests: 0 };
    if (record.freshPromptTokens >= 128 && Number.isFinite(record.prefillTokensPerSecond)) group.prefill.push(record.prefillTokensPerSecond);
    if (record.outputTokens >= 32 && Number.isFinite(record.decodeTokensPerSecond)) group.decode.push(record.decodeTokensPerSecond);
    group.requests += 1;
    groups.set(floor, group);
  }
  return [...groups.values()].sort((left, right) => left.floor - right.floor).map((group) => ({
    x: (group.floor + size / 2) / 1_000,
    requests: group.requests,
    prefill: group.prefill.length ? { count: group.prefill.length, q1: percentile(group.prefill, 0.25), median: percentile(group.prefill, 0.5), q3: percentile(group.prefill, 0.75) } : null,
    decode: group.decode.length ? { count: group.decode.length, q1: percentile(group.decode, 0.25), median: percentile(group.decode, 0.5), q3: percentile(group.decode, 0.75) } : null,
  }));
}

function rawPoint(record, x, y) {
  return { x, y, request: record.request, contextTokens: record.contextTokens, freshPromptTokens: record.freshPromptTokens, outputTokens: record.outputTokens, reusedTokens: record.reusedTokens, elapsedSeconds: record.elapsedSeconds };
}

function series(label, color, data, options = {}) {
  return {
    label,
    data: data.filter((point) => point?.y != null),
    borderColor: color,
    backgroundColor: color,
    borderWidth: options.borderWidth ?? 1.8,
    pointRadius: options.pointRadius ?? 2.5,
    pointHoverRadius: options.pointHoverRadius ?? 6,
    showLine: options.showLine ?? true,
    tension: options.tension ?? 0.12,
    fill: options.fill ?? false,
    yAxisID: options.yAxisID || "y",
    stat: options.stat,
  };
}

function bandSeries(bins, metric, label, color, backgroundColor) {
  const points = bins.filter((bin) => bin[metric]);
  return [
    series(`${label} upper quartile`, "transparent", points.map((bin) => ({ x: bin.x, y: bin[metric].q3 })), { pointRadius: 0, borderWidth: 0, stat: "band" }),
    series(`${label} IQR`, "transparent", points.map((bin) => ({ x: bin.x, y: bin[metric].q1 })), { pointRadius: 0, borderWidth: 0, fill: "-1", stat: "band" , backgroundColor }),
    series(`${label} median`, color, points.map((bin) => ({ x: bin.x, y: bin[metric].median, q1: bin[metric].q1, q3: bin[metric].q3, count: bin[metric].count, requests: bin.requests })), { pointRadius: 4, borderWidth: 2.5, stat: "median" }),
  ].map((dataset, index) => index === 1 ? { ...dataset, backgroundColor } : dataset);
}

function PerformanceChart({ title, subtitle, datasets, horizontalLabel, axes = {}, wide = false }) {
  const options = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 260, easing: "easeOutQuart" },
    interaction: { mode: "nearest", intersect: false },
    plugins: {
      legend: { labels: { color: "#443b31", usePointStyle: true, boxWidth: 8, filter: (item, data) => data.datasets[item.datasetIndex].stat !== "band" } },
      tooltip: {
        filter: (item) => item.dataset.stat !== "band",
        callbacks: {
          title(items) {
            const point = items[0]?.raw;
            if (point?.request) return `Request ${point.request} · ${formatNumber(point.contextTokens)} context tokens`;
            return `${formatNumber(point?.x, 1)}K context tokens`;
          },
          afterLabel(context) {
            const point = context.raw;
            if (context.dataset.stat === "median") return [`IQR ${formatNumber(point.q1, 1)}–${formatNumber(point.q3, 1)} tok/s`, `${point.count} qualifying requests`];
            const details = [`Elapsed ${formatDuration(point.elapsedSeconds)}`];
            if (point.freshPromptTokens != null) details.push(`Evaluated ${formatNumber(point.freshPromptTokens)} · reused ${formatNumber(point.reusedTokens)}`);
            return details;
          },
        },
      },
    },
    scales: {
      x: { type: "linear", title: { display: true, text: horizontalLabel }, grid: { color: "#ddd4c5" }, ticks: { color: "#6f675c", maxTicksLimit: 8 } },
      y: { type: "linear", beginAtZero: axes.beginAtZero ?? true, title: { display: true, text: axes.y || "Tokens per second" }, grid: { color: "#ddd4c5" }, ticks: { color: "#6f675c", maxTicksLimit: 7 } },
      ...(axes.ySecondary ? { ySecondary: { type: "linear", position: "right", beginAtZero: true, max: axes.ySecondaryMax, title: { display: true, text: axes.ySecondary }, grid: { color: "transparent" }, ticks: { color: "#6f675c", maxTicksLimit: 7 } } } : {}),
    },
  }), [axes, horizontalLabel]);
  return <section className={`log-chart ${wide ? "wide" : ""}`}><div className="log-chart-heading"><h2>{title}</h2><p>{subtitle}</p></div><div className="log-chart-frame"><Line data={{ datasets }} options={options} /></div></section>;
}

export function StrataLogDashboard() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState("");
  const [windowHours, setWindowHours] = useState("all");

  useEffect(() => { fetchStrataLog().then(setPayload).catch((requestError) => setError(requestError.message)); }, []);

  const records = useMemo(() => {
    if (!payload || windowHours === "all") return payload?.records || [];
    const cutoff = payload.metadata.durationSeconds - Number(windowHours) * 3_600;
    return payload.records.filter((record) => record.elapsedSeconds >= cutoff);
  }, [payload, windowHours]);

  const charts = useMemo(() => {
    const bins = contextBins(records);
    const qualifiedPrefill = records.filter((record) => record.freshPromptTokens >= 128);
    const qualifiedDecode = records.filter((record) => record.outputTokens >= 32);
    return {
      binned: [...bandSeries(bins, "prefill", "Prefill", COLORS.prefill, COLORS.prefillBand), ...bandSeries(bins, "decode", "Decode", COLORS.decode, COLORS.decodeBand)],
      context: [
        series("Prefill", COLORS.prefill, qualifiedPrefill.map((record) => rawPoint(record, record.contextTokens / 1_000, record.prefillTokensPerSecond)), { showLine: false }),
        series("Decode", COLORS.decode, qualifiedDecode.map((record) => rawPoint(record, record.contextTokens / 1_000, record.decodeTokensPerSecond)), { showLine: false }),
      ],
      overnight: [
        series("Prefill", COLORS.prefill, qualifiedPrefill.map((record) => rawPoint(record, record.elapsedSeconds / 3_600, record.prefillTokensPerSecond))),
        series("Decode", COLORS.decode, qualifiedDecode.map((record) => rawPoint(record, record.elapsedSeconds / 3_600, record.decodeTokensPerSecond))),
      ],
      workload: [
        series("Prompt context", COLORS.context, records.map((record) => rawPoint(record, record.elapsedSeconds / 3_600, record.contextTokens / 1_000))),
        series("Fresh suffix", COLORS.prefill, records.map((record) => rawPoint(record, record.elapsedSeconds / 3_600, record.freshPromptTokens / 1_000))),
      ],
      cache: [
        series("Cache reuse", COLORS.cache, records.map((record) => rawPoint(record, record.contextTokens / 1_000, record.cacheReusePercent)), { showLine: false }),
      ],
      latency: [
        series("Prefill", COLORS.prefill, records.map((record) => rawPoint(record, record.contextTokens / 1_000, record.prefillDurationSeconds)), { showLine: false }),
        series("Decode", COLORS.decode, records.map((record) => rawPoint(record, record.contextTokens / 1_000, record.decodeDurationSeconds)), { showLine: false }),
      ],
    };
  }, [records]);

  if (error) return <main className="log-dashboard"><a href="#">← Context Atlas</a><h1>Strata log could not be parsed</h1><p>{error}</p></main>;
  if (!payload) return <main className="log-dashboard loading-state"><div className="eyebrow">Parsing durable metrics</div><h1>Reading Strata log…</h1></main>;

  const { metadata, source, summary } = payload;
  const cacheShare = summary.reusedPromptTokens + summary.freshPromptTokens > 0 ? summary.reusedPromptTokens / (summary.reusedPromptTokens + summary.freshPromptTokens) * 100 : 0;
  return <main id="main-content" className="log-dashboard strata-dashboard">
    <nav className="log-nav"><a href="#">← Context Atlas</a><span>{source.name} · {formatNumber(source.bytes / 1_000, 1)} KB</span></nav>
    <header className="log-header"><div><div className="eyebrow">Strata · Ralph overnight record</div><h1>Through the long context</h1></div><p>{metadata.completedRequests} coding requests show throughput, prefix reuse, and latency as the working context approaches 100K tokens.</p></header>
    <dl className="log-facts">
      <div><dt>Observed runtime</dt><dd>{formatDuration(metadata.durationSeconds)}</dd></div>
      <div><dt>Peak context</dt><dd>{formatNumber(summary.peakContextTokens)} <small>tokens</small></dd></div>
      <div><dt>Median prefill</dt><dd>{formatNumber(summary.medianPrefillTokensPerSecond, 1)} <small>tok/s</small></dd></div>
      <div><dt>Median decode</dt><dd>{formatNumber(summary.medianDecodeTokensPerSecond, 1)} <small>tok/s</small></dd></div>
      <div><dt>Prefix reuse</dt><dd>{formatNumber(cacheShare, 1)}<small>% of prompt tokens</small></dd></div>
      <div><dt>Cache-hit requests</dt><dd>{formatNumber(summary.cacheHitRequests)} <small>of {metadata.completedRequests}</small></dd></div>
    </dl>
    <section className="log-controls" aria-label="Log time range"><div><strong>Analysis window</strong><span>{records.length} completed requests shown</span></div><div className="log-range" role="group" aria-label="Analysis window">
      {[{ value: "all", label: "Full run" }, { value: "6", label: "Last 6h" }, { value: "3", label: "Last 3h" }, { value: "1", label: "Last hour" }].map((option) => <button type="button" key={option.value} aria-pressed={windowHours === option.value} onClick={() => setWindowHours(option.value)}>{option.label}</button>)}
    </div></section>
    <aside className="log-interpretation-note"><strong>Robust view first.</strong><p>The headline curve bins requests into 10K-token context bands. Lines are medians and shading is the 25th–75th percentile range. Prefills under 128 fresh tokens and decodes under 32 output tokens stay out of throughput summaries but remain represented in workload and latency views.</p></aside>
    <div className="log-chart-grid">
      <PerformanceChart wide title="Typical throughput by context" subtitle="Median and interquartile range in 10K-token context bands." datasets={charts.binned} horizontalLabel="Prompt context (thousand tokens)" />
      <PerformanceChart title="Every qualifying request" subtitle="Raw observations expose variance hidden by the context-band summary." datasets={charts.context} horizontalLabel="Prompt context (thousand tokens)" />
      <PerformanceChart title="Speed through the run" subtitle="Wall-clock drift across the sustained Ralph coding session." datasets={charts.overnight} horizontalLabel="Elapsed hours" />
      <PerformanceChart title="Context trajectory" subtitle="Retained prompt context beside the newly evaluated suffix." datasets={charts.workload} horizontalLabel="Elapsed hours" axes={{ y: "Thousand tokens" }} />
      <PerformanceChart title="Prefix reuse" subtitle="Share of each prompt served from the retained Strata cache." datasets={charts.cache} horizontalLabel="Prompt context (thousand tokens)" axes={{ y: "Reused prompt (%)" }} />
      <PerformanceChart title="Request latency" subtitle="Prompt evaluation and generation duration at each context." datasets={charts.latency} horizontalLabel="Prompt context (thousand tokens)" axes={{ y: "Seconds" }} />
    </div>
    <footer className="log-provenance"><strong>{source.name}</strong><span>{new Date(metadata.startedAt).toLocaleString()} to {new Date(metadata.endedAt).toLocaleString()} · {formatNumber(metadata.parsedLines)} rows · {metadata.rejectedLines} rejected</span><code>Set STRATA_METRICS_PATH to analyze another durable metrics capture.</code></footer>
  </main>;
}