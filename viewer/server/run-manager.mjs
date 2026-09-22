import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

function stamp() {
  const date = new Date();
  const part = (value) => String(value).padStart(2, "0");
  return `${date.getFullYear()}${part(date.getMonth() + 1)}${part(date.getDate())}-${part(date.getHours())}${part(date.getMinutes())}${part(date.getSeconds())}`;
}

function slug(value) {
  return String(value).replace(/[^A-Za-z0-9._-]+/g, "--").replace(/^[-.]+|[-.]+$/g, "") || "model";
}

export function buildContextCaps(strategy, maxContext, stages = []) {
  const limit = Number(maxContext) > 0 ? Number(maxContext) : null;
  if (strategy !== "staged") return [limit];
  const caps = [...new Set(stages.map(Number).filter((value) => value > 0 && (!limit || value <= limit)))].sort((a, b) => a - b);
  if (limit && !caps.includes(limit)) caps.push(limit);
  if (!limit) caps.push(null);
  return caps;
}

export function buildDecodeVariants(model, decodeMode = "raw") {
  const supportsSpeculative = Boolean(model.speculative?.length);
  if (decodeMode === "speculative") return supportsSpeculative ? [true] : [];
  if (decodeMode === "both") return supportsSpeculative ? [false, true] : [false];
  return [false];
}

export function buildBenchmarkArguments(task) {
  const commandArguments = task.resume
    ? ["--resume", task.output, ...(task.cap ? ["--max-context", String(task.cap)] : ["--no-max-context"])]
    : ["--model", task.model.path, "--output", task.output];
  if (task.provider === "unsloth-studio") commandArguments.push("--adapter", "unsloth-llama.cpp");
  if (!task.resume && task.speculative) commandArguments.push("--speculative");
  if (!task.resume && task.speculative && task.numDraftTokens) commandArguments.push("--num-draft-tokens", String(task.numDraftTokens));
  if (!task.resume && task.cap) commandArguments.push("--max-context", String(task.cap));
  if (Number.isFinite(task.swapStopGib)) commandArguments.push("--swap-stop-gib", String(task.swapStopGib));
  return commandArguments;
}

export class RunManager {
  constructor({ launcher, runsDir }) {
    this.launcher = launcher;
    this.runsDir = runsDir;
    this.child = null;
    this.cancelRequested = false;
    this.state = { status: "idle", queue: [], current: null, log: [] };
  }

  snapshot() {
    return { ...this.state, pid: this.child?.pid || null };
  }

  _line(value) {
    const lines = String(value).split(/\r?\n/).filter(Boolean);
    this.state.log = [...this.state.log, ...lines].slice(-80);
  }

  async start({ provider = "mlx-lm", models, decodeMode = "raw", strategy = "continuous", maxContext = null, stages = [], swapStopGib = 4, numDraftTokens = null }) {
    if (this.child || this.state.status === "running") throw new Error("A benchmark job is already running.");
    if (!Array.isArray(models) || !models.length) throw new Error("Select at least one model.");
    if (!["mlx-lm", "unsloth-studio"].includes(provider)) throw new Error("Choose a supported benchmark provider.");
    if (provider === "unsloth-studio" && decodeMode !== "raw") throw new Error("Unsloth llama.cpp currently supports raw decoding only.");
    if (!["raw", "speculative", "both"].includes(decodeMode)) throw new Error("Choose raw, speculative, or both decode modes.");
    const caps = buildContextCaps(strategy, maxContext, stages);
    const runDirectories = new Map();
    const queue = [];
    for (const cap of caps) {
      for (const model of models) {
        for (const speculative of buildDecodeVariants(model, decodeMode)) {
          const variant = speculative ? "spec" : "raw";
          const key = `${model.path}:${variant}`;
          let output = runDirectories.get(key);
          const resume = Boolean(output);
          if (!output) {
            const base = path.join(this.runsDir, `${stamp()}--${slug(model.relativeName)}--${variant}`);
            output = base;
            let suffix = 2;
            while (existsSync(output)) output = `${base}-${suffix++}`;
            runDirectories.set(key, output);
          }
          queue.push({
            provider,
            model,
            cap,
            output,
            resume,
            speculative,
            numDraftTokens: Number(numDraftTokens) > 0 ? Number(numDraftTokens) : null,
            swapStopGib: Number(swapStopGib),
          });
        }
      }
    }
    if (!queue.length) throw new Error("None of the selected models supports speculative decoding.");
    this.cancelRequested = false;
    this.state = { status: "running", queue, current: null, log: [] };
    void this._drain();
    return this.snapshot();
  }

  async resume({ runDir, maxContext, swapStopGib = 4 }) {
    if (this.child || this.state.status === "running") throw new Error("A benchmark job is already running.");
    const cap = Number(maxContext) > 0 ? Number(maxContext) : null;
    this.cancelRequested = false;
    this.state = {
      status: "running",
      queue: [{
        output: runDir,
        resume: true,
        cap,
        model: { relativeName: path.basename(runDir) },
        swapStopGib: Number(swapStopGib),
      }],
      current: null,
      log: [],
    };
    void this._drain();
    return this.snapshot();
  }

  stop() {
    this.cancelRequested = true;
    this.state.status = "stopping";
    if (this.child) this.child.kill("SIGINT");
    return this.snapshot();
  }

  async _drain() {
    while (this.state.queue.length && !this.cancelRequested) {
      const task = this.state.queue[0];
      this.state.current = task;
      const commandArguments = buildBenchmarkArguments(task);
      const code = await this._spawn(commandArguments);
      if (code !== 0) {
        this.state.status = this.cancelRequested ? "stopped" : "failed";
        this.state.current = null;
        return;
      }
      this.state.queue.shift();
    }
    this.state.current = null;
    this.state.status = this.cancelRequested ? "stopped" : "completed";
  }

  _spawn(args) {
    return new Promise((resolve) => {
      this._line(`$ llm-context-bench ${args.join(" ")}`);
      const child = spawn(this.launcher, args, { cwd: path.dirname(this.launcher), env: process.env });
      this.child = child;
      child.stdout.on("data", (value) => this._line(value));
      child.stderr.on("data", (value) => this._line(value));
      child.once("error", (error) => { this._line(error.message); this.child = null; resolve(1); });
      child.once("close", (code) => { this.child = null; resolve(code ?? 1); });
    });
  }
}
