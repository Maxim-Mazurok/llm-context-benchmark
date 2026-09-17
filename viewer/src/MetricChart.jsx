import { useEffect, useMemo, useState } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";

import { legendLabelMode, seriesLabel } from "./labels.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler);

const COLORS = ["#b74228", "#254b76", "#2f766f", "#9b6b00", "#8f3f65", "#684f8e"];

function axisId(unit) {
  return `y_${unit.replace(/[^a-z0-9]/gi, "_")}`;
}

function contextForMetric(point, metric) {
  if (metric.startsWith("task_")) return point.task_prompt_tokens;
  if (metric === "prefill_tps" || metric === "prefill_duration_s") {
    return point.prefill_context_tokens;
  }
  return point.decode_context_tokens ?? point.context_tokens;
}

export default function MetricChart({
  title,
  subtitle,
  benchmarks,
  modelBenchmarks,
  metrics,
  details,
  ensureDetail,
  initialSeries,
}) {
  const [xMetric, setXMetric] = useState("context_tokens");
  const [series, setSeries] = useState(
    initialSeries.map((item, index) => ({ ...item, color: COLORS[index % COLORS.length] })),
  );
  const [editorOpen, setEditorOpen] = useState(false);

  useEffect(() => {
    for (const benchmarkId of new Set(series.map((item) => item.benchmarkId))) {
      ensureDetail(benchmarkId);
    }
  }, [ensureDetail, series]);

  const runIndex = useMemo(
    () => Object.fromEntries([...modelBenchmarks, ...benchmarks].map(
      (run) => [run.id, details[run.id] || run],
    )),
    [benchmarks, details, modelBenchmarks],
  );
  const labelMode = useMemo(() => legendLabelMode(series, runIndex), [runIndex, series]);

  const chart = useMemo(() => {
    const datasets = [];
    for (const item of series) {
      const run = details[item.benchmarkId];
      const metric = metrics[item.metric];
      if (!run || !metric) continue;
      const points = run.points
        .map((point) => {
          const x = xMetric === "context_tokens" ? contextForMetric(point, item.metric) : point[xMetric];
          const y = point[item.metric];
          return { x: Number(x), y: Number(y), cycle: point.cycle, decodeKind: point.decode_kind };
        })
        .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
      datasets.push({
        label: seriesLabel(run, metric, labelMode),
        data: points,
        borderColor: item.color,
        backgroundColor: item.color,
        borderWidth: 2.25,
        pointRadius: points.map((point) =>
          item.metric === "decode_tps" && point.decodeKind === "long" ? 7 : 3,
        ),
        pointHoverRadius: points.map((point) =>
          item.metric === "decode_tps" && point.decodeKind === "long" ? 9 : 6,
        ),
        pointBorderWidth: 0,
        tension: 0.12,
        showLine: true,
        yAxisID: axisId(metric.unit),
      });
    }
    return { datasets };
  }, [details, labelMode, metrics, series, xMetric]);

  const options = useMemo(() => {
    const units = [...new Set(series.map((item) => metrics[item.metric]?.unit).filter(Boolean))];
    const scales = {
      x: {
        type: "linear",
        title: { display: true, text: metrics[xMetric]?.label || "Actual context" },
        grid: { color: "#ddd4c5" },
        ticks: { color: "#6f675c", maxTicksLimit: 7 },
      },
    };
    units.forEach((unit, index) => {
      scales[axisId(unit)] = {
        type: "linear",
        position: index % 2 ? "right" : "left",
        beginAtZero: unit !== "GiB",
        title: { display: true, text: unit },
        grid: { color: index ? "transparent" : "#ddd4c5" },
        ticks: { color: "#6f675c", maxTicksLimit: 6 },
      };
    });
    return {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 220, easing: "easeOutQuart" },
      interaction: { mode: "nearest", intersect: false },
      plugins: {
        legend: { labels: { color: "#443b31", usePointStyle: true, boxWidth: 8 } },
        tooltip: {
          callbacks: {
            title(items) {
              const point = items[0]?.raw;
              const xLabel = metrics[xMetric]?.label || "X";
              return `${xLabel}: ${point?.x?.toLocaleString()}`;
            },
            afterLabel(context) {
              const point = context.raw;
              return point.decodeKind === "long" && context.dataset.label.includes("Decode speed")
                ? "Sustained decode probe"
                : `Cycle ${point.cycle}`;
            },
          },
        },
      },
      scales,
    };
  }, [metrics, series, xMetric]);

  const updateSeries = (index, patch) => {
    setSeries((current) => current.map((item, itemIndex) => (itemIndex === index ? { ...item, ...patch } : item)));
  };

  const addSeries = () => {
    setSeries((current) => [
      ...current,
      {
        benchmarkId: current[0]?.benchmarkId || modelBenchmarks[0]?.id || benchmarks[0]?.id,
        metric: current[0]?.metric || "decode_tps",
        color: COLORS[current.length % COLORS.length],
      },
    ]);
    setEditorOpen(true);
  };

  return (
    <section className="chart-block">
      <div className="chart-heading">
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
        <button className="quiet-button" type="button" onClick={() => setEditorOpen((open) => !open)}>
          {editorOpen ? "Hide chart controls" : "Compare or remap"}
        </button>
      </div>

      <div className={`chart-editor ${editorOpen ? "is-open" : ""}`} hidden={!editorOpen}>
        <div className="chart-editor-inner">
          <label className="axis-control">
            Horizontal axis
            <select value={xMetric} onChange={(event) => setXMetric(event.target.value)}>
              {Object.entries(metrics).map(([key, metric]) => (
                <option key={key} value={key}>{metric.label}</option>
              ))}
            </select>
          </label>
          <div className="series-editor" role="group" aria-label="Chart series">
            {series.map((item, index) => (
              <div className="series-row" key={`${index}-${item.benchmarkId}-${item.metric}`}>
                <input
                  aria-label={`Series ${index + 1} color`}
                  className="color-input"
                  type="color"
                  value={item.color}
                  onChange={(event) => updateSeries(index, { color: event.target.value })}
                />
                <label>
                  Benchmark
                  <select
                    value={item.benchmarkId}
                    onChange={(event) => updateSeries(index, { benchmarkId: event.target.value })}
                  >
                    <optgroup label="Models · averaged">
                      {modelBenchmarks.map((benchmark) => (
                        <option key={benchmark.id} value={benchmark.id}>
                          {benchmark.modelName} · {benchmark.runCount} capacity run{benchmark.runCount === 1 ? "" : "s"}
                        </option>
                      ))}
                    </optgroup>
                    {modelBenchmarks.map((modelBenchmark) => {
                      const modelRuns = benchmarks.filter(
                        (benchmark) => benchmark.modelName === modelBenchmark.modelName,
                      );
                      if (!modelRuns.length) return null;
                      return <optgroup key={modelBenchmark.id} label={`${modelBenchmark.modelName} · individual runs`}>
                        {modelRuns.map((benchmark) => (
                          <option key={benchmark.id} value={benchmark.id}>
                            {benchmark.kind === "useful-task-run" ? "Useful tasks" : "Capacity"} · {new Date(benchmark.runAt).toLocaleString()}
                          </option>
                        ))}
                      </optgroup>;
                    })}
                  </select>
                </label>
                <label>
                  Metric
                  <select value={item.metric} onChange={(event) => updateSeries(index, { metric: event.target.value })}>
                    {Object.entries(metrics).filter(([key]) => key !== "context_tokens").map(([key, metric]) => (
                      <option key={key} value={key}>{metric.label}</option>
                    ))}
                  </select>
                </label>
                <button
                  className="remove-button"
                  type="button"
                  disabled={series.length === 1}
                  onClick={() => setSeries((current) => current.filter((_, itemIndex) => itemIndex !== index))}
                >
                  Remove series
                </button>
              </div>
            ))}
          </div>
          <button className="add-button" type="button" onClick={addSeries}>Add another benchmark or metric</button>
        </div>
      </div>

      <div className="chart-frame">
        {chart.datasets.length ? (
          <Line
            aria-label={`${title}. Interactive chart; use the comparison controls to change its series and axes.`}
            role="img"
            data={chart}
            options={options}
          />
        ) : <div className="chart-loading">Reading benchmark points…</div>}
      </div>
      {series.some((item) => item.metric === "decode_tps") && (
        <p className="chart-footnote"><span className="large-dot" /> Larger dots mark sustained decode probes; all points share one decode line.</p>
      )}
    </section>
  );
}
