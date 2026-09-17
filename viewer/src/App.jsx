import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  fetchBenchmark, fetchBenchmarks, fetchModels, fetchRunner,
  resumeBenchmark, startRunner, stopRunner,
} from "./api.js";
import MetricChart, { chartColor } from "./MetricChart.jsx";

function formatNumber(value, digits = 0) {
  if (value == null || !Number.isFinite(Number(value))) return "—";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatRunDate(value) {
  return new Date(value).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function RunnerPanel({ models, runner, onRunnerChange, onRefresh }) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState([]);
  const [strategy, setStrategy] = useState("staged");
  const [decodeMode, setDecodeMode] = useState("raw");
  const [maxContext, setMaxContext] = useState("");
  const [stages, setStages] = useState("8192, 16384, 32768, 65536, 131072");
  const [swapStopGib, setSwapStopGib] = useState("4");
  const [numDraftTokens, setNumDraftTokens] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const active = ["running", "stopping"].includes(runner?.status);
  const submit = async () => {
    setBusy(true); setMessage("");
    try {
      const chosen = models.filter((model) => selected.includes(model.path));
      const next = await startRunner({
        models: chosen, decodeMode, strategy,
        maxContext: Number(maxContext) || null,
        stages: stages.split(/[ ,]+/).map(Number).filter(Boolean),
        swapStopGib: Number(swapStopGib),
        numDraftTokens: Number(numDraftTokens) || null,
      });
      onRunnerChange(next); setOpen(true);
    } catch (error) { setMessage(error.message); } finally { setBusy(false); }
  };
  const stop = async () => { setBusy(true); try { onRunnerChange(await stopRunner()); } catch (error) { setMessage(error.message); } finally { setBusy(false); } };
  return (
    <section className={`runner-panel ${active ? "is-active" : ""}`}>
      <button className="runner-summary" type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open}>
        <span><strong>{active ? (runner.status === "stopping" ? "Saving checkpoint…" : "Benchmark running") : "Run benchmarks"}</strong><small>{active ? `${runner.current?.model?.relativeName || runner.current?.output}${runner.current?.speculative ? " · speculative" : " · raw"}` : "Launch, stop, and resume from the browser"}</small></span>
        <span className="runner-toggle">{open ? "Close" : "Configure"}</span>
      </button>
      <div className={`runner-body ${open ? "is-open" : ""}`}><div className="runner-body-inner">
        {!active && <>
          <div className="runner-toolbar"><span>{selected.length} of {models.length} models selected</span><button type="button" onClick={() => setSelected(models.map((model) => model.path))}>Select all</button><button type="button" onClick={() => setSelected([])}>Clear</button></div>
          <div className="model-picker">
            {models.map((model) => <label key={model.path}><input type="checkbox" checked={selected.includes(model.path)} onChange={(event) => setSelected((current) => event.target.checked ? [...current, model.path] : current.filter((path) => path !== model.path))} /><span><strong>{model.relativeName}</strong><small>{model.speculative?.length ? model.speculative[0].description : "Raw decoding only"}</small></span></label>)}
          </div>
          <div className="run-options">
            <fieldset><legend>Schedule</legend><label><input type="radio" name="strategy" checked={strategy === "staged"} onChange={() => setStrategy("staged")} /> Round-robin stages</label><label><input type="radio" name="strategy" checked={strategy === "continuous"} onChange={() => setStrategy("continuous")} /> Finish each model</label></fieldset>
            <label>Final context target <span className="optional">optional</span><input type="number" min="1" value={maxContext} placeholder="Model/runtime limit" onChange={(event) => setMaxContext(event.target.value)} /><small>Leave empty to continue until the model or runtime stops.</small></label>
            {strategy === "staged" && <label>Context stages<input type="text" value={stages} onChange={(event) => setStages(event.target.value)} /></label>}
            <label>Stop after added swap<input type="number" min="0" step="0.5" value={swapStopGib} onChange={(event) => setSwapStopGib(event.target.value)} /><small>GiB peak growth; raise this to allow heavily swapped runs.</small></label>
            <fieldset className="decode-mode"><legend>Decode variants</legend><label><input type="radio" name="decodeMode" checked={decodeMode === "raw"} onChange={() => setDecodeMode("raw")} /> Raw only</label><label><input type="radio" name="decodeMode" checked={decodeMode === "speculative"} onChange={() => setDecodeMode("speculative")} /> Speculative only</label><label><input type="radio" name="decodeMode" checked={decodeMode === "both"} onChange={() => setDecodeMode("both")} /> Raw + speculative</label><small>“Both” creates separate comparable runs. Models without a drafter still run raw.</small></fieldset>
            {decodeMode !== "raw" && <label>Speculative draft blocks <span className="optional">optional</span><input type="number" min="1" value={numDraftTokens} placeholder="OMLX setting, otherwise 3" onChange={(event) => setNumDraftTokens(event.target.value)} /><small>Maximum adaptive MTP draft depth; an explicit value overrides OMLX.</small></label>}
          </div>
          <button className="start-button" type="button" disabled={busy || !selected.length} onClick={submit}>Start benchmark queue</button>
        </>}
        {active && <div className="active-run"><div><span className="pulse" />Stage target <strong>{runner.current?.cap ? `${formatNumber(runner.current.cap)} tokens` : "model limit"}</strong> · {runner.queue?.length || 0} steps remaining</div><button className="stop-button" type="button" disabled={busy || runner.status === "stopping"} onClick={stop}>Stop gracefully</button></div>}
        {!!runner?.log?.length && <pre className="runner-log">{runner.log.slice(-14).join("\n")}</pre>}
        {message && <p className="runner-message">{message}</p>}
      </div></div>
    </section>
  );
}

function ModelComparison({ benchmarks, modelBenchmarks, metrics, details, ensureDetail }) {
  const capacityModels = useMemo(
    () => modelBenchmarks.filter((benchmark) => benchmark.runCount > 0),
    [modelBenchmarks],
  );
  const [selectedIds, setSelectedIds] = useState(() => capacityModels.map((benchmark) => benchmark.id));
  const knownIds = useRef(new Set(capacityModels.map((benchmark) => benchmark.id)));

  useEffect(() => {
    const added = capacityModels
      .map((benchmark) => benchmark.id)
      .filter((id) => !knownIds.current.has(id));
    if (added.length) setSelectedIds((current) => [...current, ...added]);
    knownIds.current = new Set(capacityModels.map((benchmark) => benchmark.id));
  }, [capacityModels]);

  const selected = new Set(selectedIds);
  const seriesFor = (metric) => capacityModels
    .map((benchmark, index) => ({ benchmarkId: benchmark.id, metric, color: chartColor(index) }))
    .filter((item) => selected.has(item.benchmarkId));
  const toggle = (id, checked) => setSelectedIds((current) => (
    checked ? [...new Set([...current, id])] : current.filter((item) => item !== id)
  ));

  return (
    <section className="comparison-section" aria-labelledby="comparison-title">
      <div className="comparison-heading">
        <div>
          <div className="eyebrow">Across the field</div>
          <h2 id="comparison-title">Compare every model</h2>
          <p>One selection controls every chart below. Aggregated model lines average repeated runs at matching contexts.</p>
        </div>
        <div className="comparison-actions">
          <span>{selectedIds.length} of {capacityModels.length} selected</span>
          <button type="button" onClick={() => setSelectedIds(capacityModels.map((model) => model.id))}>Select all</button>
          <button type="button" onClick={() => setSelectedIds([])}>Clear</button>
        </div>
      </div>
      <div className="comparison-picker" role="group" aria-label="Models shown in comparison charts">
        {capacityModels.map((model, index) => (
          <label key={model.id}>
            <input
              type="checkbox"
              checked={selected.has(model.id)}
              onChange={(event) => toggle(model.id, event.target.checked)}
            />
            <span className="model-swatch" style={{ "--series-color": chartColor(index) }} aria-hidden="true" />
            <span>{model.modelName}</span>
          </label>
        ))}
      </div>
      <div className="comparison-charts">
        <MetricChart
          title="Prefill speed"
          subtitle="Append-only processing throughput by actual context"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[]} seriesOverride={seriesFor("prefill_tps")} editable={false}
        />
        <MetricChart
          title="Decode speed"
          subtitle="Generation throughput by actual context"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[]} seriesOverride={seriesFor("decode_tps")} editable={false}
        />
        <MetricChart
          title="MLX active memory"
          subtitle="Maximum active MLX allocation at each tested context"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[]} seriesOverride={seriesFor("mlx_active_gib")} editable={false}
        />
        <MetricChart
          title="Swap growth"
          subtitle="Peak swap growth since model load"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[]} seriesOverride={seriesFor("swap_growth_gib")} editable={false}
        />
      </div>
    </section>
  );
}

function RunLedger({ benchmarks, modelBenchmarks, metrics, details, ensureDetail, models, runner, onRunnerChange, onSelect, onRefresh, refreshing }) {
  const [query, setQuery] = useState("");
  const [ledgerMode, setLedgerMode] = useState("models");
  const entries = ledgerMode === "models" ? modelBenchmarks : benchmarks;
  const filtered = entries.filter((run) =>
    `${run.modelName} ${run.runAt} ${run.stopReason}`.toLowerCase().includes(query.toLowerCase()),
  );
  return (
    <main id="main-content" className="page-shell">
      <header className="masthead">
        <div className="eyebrow">Local inference field notes</div>
        <h1>Context Atlas</h1>
        <p className="lede">See where local models slow down, consume memory, and begin to swap as their live context grows.</p>
      </header>
      <RunnerPanel models={models} runner={runner} onRunnerChange={onRunnerChange} onRefresh={onRefresh} />
      <ModelComparison
        benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics}
        details={details} ensureDetail={ensureDetail}
      />
      <div className="ledger-tools">
        <label className="search-field">
          Find {ledgerMode === "models" ? "a model" : "a benchmark"}
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder={ledgerMode === "models" ? "Model name" : "Model, date, or stop reason"} />
        </label>
        <div className="ledger-actions">
          <div className="ledger-mode" role="group" aria-label="Ledger view">
            <button type="button" aria-pressed={ledgerMode === "models"} onClick={() => setLedgerMode("models")}>Models</button>
            <button type="button" aria-pressed={ledgerMode === "runs"} onClick={() => setLedgerMode("runs")}>Individual runs</button>
          </div>
          <button className="quiet-button" type="button" onClick={onRefresh} disabled={refreshing}>
            {refreshing ? "Refreshing runs…" : "Refresh runs"}
          </button>
        </div>
      </div>
      <section className="ledger" aria-label={ledgerMode === "models" ? "Benchmarked models" : "Benchmark runs"}>
        <div className="ledger-header" aria-hidden="true">
          <span>{ledgerMode === "models" ? "Benchmarked model" : "Model and local run time"}</span><span>Maximum context</span><span>Peak MLX</span><span>Swap growth</span><span>{ledgerMode === "models" ? "Evidence" : "Result"}</span>
        </div>
        {filtered.map((run, index) => (
          <button className="ledger-row" style={{ "--i": Math.min(index, 8) }} type="button" key={run.id} onClick={() => onSelect(run.id)}>
            <span className="run-identity"><strong>{run.modelName}</strong><small>{run.kind === "model" ? `Latest evidence ${formatRunDate(run.runAt)}` : formatRunDate(run.runAt)}</small></span>
            <span data-label="Maximum context"><strong>{formatNumber(run.maxContextTokens)}</strong><small>tokens</small></span>
            <span data-label="Peak MLX"><strong>{formatNumber(run.peakMlxActiveGib, 2)}</strong><small>GiB active</small></span>
            <span data-label="Swap growth"><strong>{formatNumber(run.peakSwapGrowthGib, 2)}</strong><small>GiB</small></span>
            <span data-label={run.kind === "model" ? "Evidence" : "Result"}><strong className={`stop-reason status-${run.status}`}>{run.kind === "model" ? `${run.runCount} capacity run${run.runCount === 1 ? "" : "s"}` : run.status === "running" ? "running now" : run.stopReason.replaceAll("_", " ")}</strong><small>{run.kind === "model" ? `${run.observationCount} useful-task observation${run.observationCount === 1 ? "" : "s"}` : `${formatNumber(run.finalDecodeTps, 1)} tok/s final decode`}</small></span>
            <span className="row-arrow" aria-hidden="true">→</span>
          </button>
        ))}
        {!filtered.length && <div className="empty-state">No matching {ledgerMode === "models" ? "benchmarked models" : "runs"}. Try another model name or refresh the list.</div>}
      </section>
    </main>
  );
}

function DetailView({ benchmark, benchmarks, modelBenchmarks, metrics, details, ensureDetail, onBack, runner, onRunnerChange }) {
  const defaultId = benchmark.id;
  const aggregate = benchmark.kind === "model";
  const hasCapacity = aggregate ? benchmark.runCount > 0 : benchmark.kind === "capacity-run";
  const hasUsefulTasks = aggregate ? benchmark.observationCount > 0 : benchmark.kind === "useful-task-run";
  const [resumeTarget, setResumeTarget] = useState("");
  const [resumeError, setResumeError] = useState("");
  const resume = async () => { try { onRunnerChange(await resumeBenchmark(benchmark.id, Number(resumeTarget) || null)); setResumeError(""); } catch (error) { setResumeError(error.message); } };
  return (
    <main id="main-content" className="page-shell detail-page">
      <button className="back-button" type="button" onClick={onBack}>← All benchmarked models</button>
      <header className="detail-header">
        <div>
          <div className="eyebrow">{aggregate ? "Averaged across all matching evidence" : formatRunDate(benchmark.runAt)}</div>
          <h1>{benchmark.modelName}</h1>
          <p className="model-path">{benchmark.model}</p>
        </div>
        <dl className="run-facts">
          <div><dt>Maximum tested</dt><dd>{formatNumber(benchmark.maxContextTokens)} <small>tokens</small></dd></div>
          <div><dt>Peak MLX active</dt><dd>{formatNumber(benchmark.peakMlxActiveGib, 2)} <small>GiB</small></dd></div>
          <div><dt>Peak total swap</dt><dd>{formatNumber(benchmark.peakSwapUsedGib, 2)} <small>GiB</small></dd></div>
          <div><dt>Peak swap growth</dt><dd>{formatNumber(benchmark.peakSwapGrowthGib, 2)} <small>GiB</small></dd></div>
          <div><dt>{aggregate ? "Capacity runs" : "Worst pressure"}</dt><dd className="reason-value">{aggregate ? formatNumber(benchmark.runCount) : benchmark.worstMemoryPressure || "—"}</dd></div>
          <div><dt>{aggregate ? "Useful-task observations" : "Stopped"}</dt><dd className="reason-value">{aggregate ? formatNumber(benchmark.observationCount) : benchmark.stopReason.replaceAll("_", " ")}</dd></div>
        </dl>
      </header>
      {benchmark.resumable && !["running", "stopping"].includes(runner?.status) && <section className="resume-strip"><span><strong>Continue this cache</strong><small>The model reloads and reconstructs the last completed context checkpoint.</small></span><label>Final target <span className="optional">optional</span><input type="number" min={benchmark.maxContextTokens + 1} value={resumeTarget} placeholder="Model/runtime limit" onChange={(event) => setResumeTarget(event.target.value)} /></label><button type="button" onClick={resume}>Resume benchmark</button>{resumeError && <small className="runner-message">{resumeError}</small>}</section>}
      <aside className="measurement-note">
        <strong>{aggregate ? "Model aggregate." : "Period-local measurements."}</strong> {aggregate ? "Capacity values are averaged only where runs share the same actual context. Useful tasks remain independent requests and appear in their own chart." : "Each prefill point measures only that append phase—not elapsed time since the run began. Peak swap retains brief spikes that cycle-end readings miss. Memory pressure can oscillate independently as macOS compresses, evicts, pages, and reclaims memory."}
        {benchmark.measurement?.memory && <small>{benchmark.measurement.memory}</small>}
      </aside>
      <div className="chart-stack">
        {hasCapacity && <MetricChart
          key={`${defaultId}-prefill`}
          title="Prefill scaling"
          subtitle="Processing speed for each individual context append"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[{ benchmarkId: defaultId, metric: "prefill_tps" }]}
        />}
        {hasCapacity && <MetricChart
          key={`${defaultId}-decode`}
          title="Decode scaling"
          subtitle="One continuous line; sustained probes use larger points"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[{ benchmarkId: defaultId, metric: "decode_tps" }]}
        />}
        {hasCapacity && <MetricChart
          key={`${defaultId}-memory`}
          title="Memory behavior"
          subtitle="Process-specific MLX allocation beside system pressure indicators"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[
            { benchmarkId: defaultId, metric: "mlx_active_gib" },
            { benchmarkId: defaultId, metric: "system_available_gib" },
            { benchmarkId: defaultId, metric: "compressed_gib" },
            { benchmarkId: defaultId, metric: "swap_total_gib" },
            { benchmarkId: defaultId, metric: "swap_growth_gib" },
          ]}
        />}
        {hasUsefulTasks && <MetricChart
          key={`${defaultId}-useful-tasks`}
          title="Useful-task behavior"
          subtitle="Independent task requests by real prompt length"
          benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail}
          initialSeries={[
            { benchmarkId: defaultId, metric: "task_generation_tps" },
            { benchmarkId: defaultId, metric: "task_score" },
          ]}
        />}
      </div>
    </main>
  );
}

export default function App() {
  const [benchmarks, setBenchmarks] = useState([]);
  const [modelBenchmarks, setModelBenchmarks] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [details, setDetails] = useState({});
  const [selectedId, setSelectedId] = useState(() => decodeURIComponent(location.hash.replace("#run=", "")) || null);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(true);
  const [models, setModels] = useState([]);
  const [runner, setRunner] = useState({ status: "idle", queue: [], log: [] });
  const pendingDetails = useRef(new Map());

  const refresh = useCallback(async () => {
    setRefreshing(true);
    try {
      const payload = await fetchBenchmarks();
      setBenchmarks(payload.benchmarks);
      setModelBenchmarks(payload.modelBenchmarks || []);
      setMetrics(payload.metrics);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setRefreshing(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => { fetchModels().then((payload) => setModels(payload.models || [])).catch((requestError) => setError(requestError.message)); }, []);
  useEffect(() => {
    const poll = async () => {
      try {
        const next = await fetchRunner(); setRunner(next);
        await refresh();
        if (selectedId) {
          const detail = await fetchBenchmark(selectedId).catch(() => null);
          if (detail) setDetails((current) => ({ ...current, [selectedId]: detail }));
        }
      } catch { /* the main refresh surface reports persistent failures */ }
    };
    poll(); const timer = setInterval(poll, 2000); return () => clearInterval(timer);
  }, [refresh, selectedId]);
  useEffect(() => {
    const onHashChange = () => setSelectedId(decodeURIComponent(location.hash.replace("#run=", "")) || null);
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const ensureDetail = useCallback(async (id) => {
    if (!id || details[id]) return details[id];
    if (pendingDetails.current.has(id)) return pendingDetails.current.get(id);
    const request = (async () => { try {
      const detail = await fetchBenchmark(id);
      setDetails((current) => ({ ...current, [id]: detail }));
      return detail;
    } catch (requestError) {
      setError(requestError.message);
      return null;
    } finally {
      pendingDetails.current.delete(id);
    } })();
    pendingDetails.current.set(id, request);
    return request;
  }, [details]);

  useEffect(() => { if (selectedId) ensureDetail(selectedId); }, [ensureDetail, selectedId]);

  const selected = useMemo(
    () => details[selectedId]
      || modelBenchmarks.find((benchmark) => benchmark.id === selectedId)
      || benchmarks.find((benchmark) => benchmark.id === selectedId),
    [benchmarks, details, modelBenchmarks, selectedId],
  );

  const selectRun = (id) => { location.hash = `run=${encodeURIComponent(id)}`; };
  const showList = () => { history.pushState(null, "", location.pathname); setSelectedId(null); };

  if (error) {
    return <main className="page-shell"><div className="error-state"><h1>Benchmark data could not be loaded</h1><p>{error}</p><button onClick={refresh}>Try loading again</button></div></main>;
  }
  if (!benchmarks.length && refreshing) {
    return <main className="page-shell loading-state"><div className="eyebrow">Indexing local runs</div><h1>Building the atlas…</h1></main>;
  }
  if (selectedId && selected) {
    return <DetailView benchmark={selected} benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail} onBack={showList} runner={runner} onRunnerChange={setRunner} />;
  }
  return <RunLedger benchmarks={benchmarks} modelBenchmarks={modelBenchmarks} metrics={metrics} details={details} ensureDetail={ensureDetail} models={models} runner={runner} onRunnerChange={setRunner} onSelect={selectRun} onRefresh={refresh} refreshing={refreshing} />;
}
