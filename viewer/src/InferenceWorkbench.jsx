import { useDeferredValue, useEffect, useRef, useState } from "react";
import {
  CategoryScale,
  Chart as ChartJS,
  LinearScale,
  LineElement,
  PointElement,
  Tooltip,
} from "chart.js";
import { Line } from "react-chartjs-2";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { fetchModels, streamInference, uploadInferenceDocument } from "./api.js";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip);

const numberFormatter = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });

function formatEta(seconds, unlimited = false) {
  if (unlimited) return "Model decides";
  if (!Number.isFinite(seconds)) return "Calculating";
  if (seconds < 1) return "< 1 sec";
  const roundedSeconds = Math.ceil(seconds);
  if (roundedSeconds < 60) return `${roundedSeconds} sec`;
  const minutes = Math.floor(roundedSeconds / 60);
  const remainingSeconds = roundedSeconds % 60;
  return `${minutes}m ${remainingSeconds}s`;
}

function SpeedChart({ title, points, color }) {
  const chartPoints = points.filter((point) => Number.isFinite(point.y) && point.y > 0);
  const data = {
    datasets: [{
      data: chartPoints,
      borderColor: color,
      backgroundColor: color,
      borderWidth: 2,
      pointRadius: chartPoints.length > 40 ? 0 : 2,
      pointHoverRadius: 5,
      tension: 0.16,
    }],
  };
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "Context tokens" },
        grid: { color: "#d7d0c2" },
      },
      y: {
        beginAtZero: true,
        title: { display: true, text: "Tokens / second" },
        grid: { color: "#d7d0c2" },
      },
    },
  };
  return <section className="live-chart"><h2>{title}</h2><div>{chartPoints.length ? <Line data={data} options={options} /> : <span>Waiting for a stable speed sample</span>}</div></section>;
}

export function InferenceWorkbench() {
  const [models, setModels] = useState([]);
  const [model, setModel] = useState("");
  const [file, setFile] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [maxOutputTokens, setMaxOutputTokens] = useState("1024");
  const [outputUnlimited, setOutputUnlimited] = useState(false);
  const [reasoningEnabled, setReasoningEnabled] = useState(false);
  const [maxReasoningTokens, setMaxReasoningTokens] = useState("2048");
  const [reasoningUnlimited, setReasoningUnlimited] = useState(true);
  const [phase, setPhase] = useState("idle");
  const [uploadProgress, setUploadProgress] = useState(0);
  const [promptTokens, setPromptTokens] = useState(0);
  const [processedTokens, setProcessedTokens] = useState(0);
  const [generatedTokens, setGeneratedTokens] = useState(0);
  const [reasoningTokens, setReasoningTokens] = useState(0);
  const [outputTokens, setOutputTokens] = useState(0);
  const [prefillEta, setPrefillEta] = useState(null);
  const [decodeEta, setDecodeEta] = useState(null);
  const [decodeStage, setDecodeStage] = useState("answer");
  const [finishReason, setFinishReason] = useState("");
  const [prefillPoints, setPrefillPoints] = useState([]);
  const [decodePoints, setDecodePoints] = useState([]);
  const [output, setOutput] = useState("");
  const [reasoning, setReasoning] = useState("");
  const [error, setError] = useState("");
  const abortController = useRef(null);
  const deferredOutput = useDeferredValue(output);
  const deferredReasoning = useDeferredValue(reasoning);

  useEffect(() => {
    fetchModels().then((payload) => {
      const gemmaModels = (payload.models || []).filter((item) => /gemma-4.*12b/i.test(item.relativeName));
      setModels(gemmaModels);
      setModel(gemmaModels[0]?.path || "");
    }).catch((requestError) => setError(requestError.message));
    return () => abortController.current?.abort();
  }, []);

  const running = !["idle", "complete", "error", "stopped"].includes(phase);
  const submit = async (event) => {
    event.preventDefault();
    if (!file || !model || !prompt.trim()) return;
    setError(""); setOutput(""); setReasoning(""); setPromptTokens(0); setProcessedTokens(0); setGeneratedTokens(0);
    setReasoningTokens(0); setOutputTokens(0); setPrefillEta(null); setDecodeEta(null); setFinishReason("");
    setPrefillPoints([]); setDecodePoints([]); setUploadProgress(0); setPhase("uploading");
    abortController.current = new AbortController();
    try {
      const { uploadId } = await uploadInferenceDocument(file, setUploadProgress);
      setPhase("loading");
      let currentPromptTokens = 0;
      await streamInference({
        uploadId,
        model,
        prompt,
        maxOutputTokens: outputUnlimited ? null : Number(maxOutputTokens),
        reasoningEnabled,
        maxReasoningTokens: reasoningEnabled && !reasoningUnlimited ? Number(maxReasoningTokens) : null,
      }, (message) => {
        if (message.type === "status") setPhase(message.phase);
        if (message.type === "prompt") {
          currentPromptTokens = message.totalTokens;
          setPromptTokens(message.totalTokens);
          setPhase("prefill");
        }
        if (message.type === "prefill") {
          setProcessedTokens(message.processedTokens);
          setPrefillEta(message.etaSeconds);
          setPrefillPoints((current) => [...current, { x: message.processedTokens, y: message.tokensPerSecond }]);
        }
        if (message.type === "decode") {
          setPhase("decode"); setGeneratedTokens(message.generatedTokens);
          setReasoningTokens(message.reasoningTokens); setOutputTokens(message.outputTokens);
          setDecodeEta(message.etaSeconds); setDecodeStage(message.stage);
          setReasoning((current) => current + message.reasoning);
          setOutput((current) => current + message.token);
          if (Number.isFinite(message.tokensPerSecond)) {
            setDecodePoints((current) => [...current, { x: currentPromptTokens + message.generatedTokens, y: message.tokensPerSecond }]);
          }
        }
        if (message.type === "text") {
          setReasoning((current) => current + message.reasoning);
          setOutput((current) => current + message.token);
        }
        if (message.type === "complete") {
          setGeneratedTokens(message.generatedTokens); setReasoningTokens(message.reasoningTokens); setOutputTokens(message.outputTokens);
          setFinishReason(message.finishReason); setDecodeEta(0); setPhase("complete");
        }
        if (message.type === "error") throw new Error(message.message);
      }, abortController.current.signal);
    } catch (requestError) {
      if (requestError.name === "AbortError") setPhase("stopped");
      else { setError(requestError.message); setPhase("error"); }
    }
  };

  return <main id="main-content" className="workbench-shell">
    <nav className="workbench-nav"><a href="#">Context Atlas</a><span>Gemma document workbench</span></nav>
    <header className="workbench-header">
      <div className="eyebrow">Raw decoding · pageable memory</div>
      <h1>Read the whole thing.</h1>
      <p>Attach a large text file, give Gemma one instruction, and watch context processing turn into streamed output.</p>
    </header>

    <form className="workbench-grid" onSubmit={submit}>
      <section className="document-controls">
        <label className={`file-drop ${file ? "has-file" : ""}`}>
          <input type="file" accept="text/*,.txt,.md,.csv,.json,.jsonl,.log" onChange={(event) => setFile(event.target.files[0] || null)} />
          <strong>{file ? file.name : "Drop or choose a text file"}</strong>
          <span>{file ? `${numberFormatter.format(file.size / 1_048_576)} MiB` : "Text, Markdown, CSV, JSONL, or logs"}</span>
        </label>
        <label>Model<select value={model} onChange={(event) => setModel(event.target.value)} disabled={running}>
          {!models.length && <option value="">No installed Gemma 4 12B model found</option>}
          {models.map((item) => <option key={item.path} value={item.path}>{item.relativeName}</option>)}
        </select></label>
        <label>Instruction<textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Summarize the document and identify unresolved decisions." rows="7" /></label>
        <fieldset className="generation-options">
          <legend>Generation</legend>
          <label className="check-option"><input type="checkbox" checked={outputUnlimited} onChange={(event) => setOutputUnlimited(event.target.checked)} /> Unlimited output</label>
          <label>Maximum output tokens<input type="number" min="1" value={maxOutputTokens} disabled={outputUnlimited} onChange={(event) => setMaxOutputTokens(event.target.value)} /></label>
          <label className="check-option"><input type="checkbox" checked={reasoningEnabled} onChange={(event) => setReasoningEnabled(event.target.checked)} /> Show reasoning</label>
          {reasoningEnabled && <>
            <label className="check-option"><input type="checkbox" checked={reasoningUnlimited} onChange={(event) => setReasoningUnlimited(event.target.checked)} /> Unlimited reasoning</label>
            <label>Maximum reasoning tokens<input type="number" min="1" value={maxReasoningTokens} disabled={reasoningUnlimited} onChange={(event) => setMaxReasoningTokens(event.target.value)} /><small>Stops generation if reached before final answer.</small></label>
          </>}
        </fieldset>
        <div className="workbench-actions">
          <button type="submit" disabled={running || !file || !model || !prompt.trim()}>Run inference</button>
          {running && <button type="button" className="cancel-button" onClick={() => abortController.current?.abort()}>Stop</button>}
        </div>
        {error && <p className="runner-message">{error}</p>}
      </section>

      <section className="output-panel" aria-live="polite">
        <div className="phase-line"><span className={`phase-dot phase-${phase}`} />{phase === "idle" ? "Ready" : phase}{finishReason ? ` · ${finishReason.replaceAll("_", " ")}` : ""}</div>
        <div className="live-metrics">
          <div><span>Prefill</span><strong>{numberFormatter.format(processedTokens)} / {numberFormatter.format(promptTokens)}</strong><small>ETA {formatEta(prefillEta)}</small></div>
          <div><span>Reasoning</span><strong>{numberFormatter.format(reasoningTokens)}</strong><small>{decodeStage === "reasoning" ? `ETA ${formatEta(decodeEta, reasoningUnlimited)}` : "Complete"}</small></div>
          <div><span>Answer</span><strong>{numberFormatter.format(outputTokens)}</strong><small>{decodeStage === "answer" ? `ETA ${formatEta(decodeEta, outputUnlimited)}` : "Waiting"}</small></div>
          <div><span>Total decoded</span><strong>{numberFormatter.format(generatedTokens)}</strong><small>Upload {numberFormatter.format(uploadProgress * 100)}%</small></div>
        </div>
        {reasoningEnabled && (reasoning || phase === "decode") && <details className="reasoning-output" open><summary>Reasoning · {numberFormatter.format(reasoningTokens)} tokens</summary><pre>{deferredReasoning || "Waiting for reasoning…"}</pre></details>}
        <div className={`model-output markdown-output ${output ? "has-output" : ""}`}>
          {output ? <ReactMarkdown remarkPlugins={[remarkGfm]}>{deferredOutput}</ReactMarkdown> : "Model output will stream here."}
        </div>
      </section>
    </form>

    <div className="live-charts">
      <SpeedChart title="Prefill speed" points={prefillPoints} color="#b7472a" />
      <SpeedChart title="Decode speed" points={decodePoints} color="#245e68" />
    </div>
  </main>;
}