import http from "node:http";
import os from "node:os";
import path from "node:path";
import { createWriteStream, existsSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import { execFile, spawn } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { randomUUID } from "node:crypto";
import { pipeline } from "node:stream/promises";

import express from "express";

import { buildModelBenchmarks, getBenchmark, listBenchmarks, METRICS } from "./data.mjs";
import { RunManager } from "./run-manager.mjs";

const serverDir = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(serverDir, "../..");
const viewerRoot = path.join(projectRoot, "viewer");
const runsDir = path.resolve(process.env.BENCHMARK_RUNS_DIR || path.join(projectRoot, "runs"));
const launcher = path.join(projectRoot, "bin", "llm-context-bench");
const inferenceLauncher = path.join(projectRoot, "bin", "llm-context-infer");
const uploadDirectory = path.join(os.tmpdir(), "llm-context-benchmark-uploads");
const omlxModelsDirectory = path.join(os.homedir(), ".omlx", "models");
const uploads = new Map();
const runManager = new RunManager({ launcher, runsDir });
const execFileAsync = promisify(execFile);
const production = process.argv.includes("--production") || process.env.NODE_ENV === "production";

const app = express();
const httpServer = http.createServer(app);
app.use(express.json());

app.get("/api/health", (_request, response) => {
  response.json({ ok: true, runsDir, runner: runManager.snapshot() });
});

app.get("/api/models", async (_request, response, next) => {
  try {
    const { stdout } = await execFileAsync(launcher, ["--list-omlx-models", "--json"], { maxBuffer: 10 * 1024 * 1024 });
    response.json(JSON.parse(stdout));
  } catch (error) { next(error); }
});

app.post("/api/inference/files", async (request, response, next) => {
  const uploadId = randomUUID();
  const documentPath = path.join(uploadDirectory, uploadId);
  try {
    await mkdir(uploadDirectory, { recursive: true });
    await pipeline(request, createWriteStream(documentPath, { flags: "wx" }));
    uploads.set(uploadId, { documentPath, fileName: request.get("x-file-name") || "document.txt" });
    response.status(201).json({ uploadId });
  } catch (error) {
    await rm(documentPath, { force: true });
    next(error);
  }
});

app.post("/api/inference/generate", (request, response) => {
  const { uploadId, model, prompt, maxOutputTokens, reasoningEnabled, maxReasoningTokens } = request.body;
  const upload = uploads.get(uploadId);
  const modelPath = typeof model === "string" ? path.resolve(model) : "";
  if (!upload) return response.status(404).json({ error: "Uploaded document was not found." });
  if (!modelPath.startsWith(`${omlxModelsDirectory}${path.sep}`) || !modelPath.toLowerCase().includes("gemma-4-12")) {
    return response.status(400).json({ error: "Select an installed Gemma 4 12B model." });
  }
  if (typeof prompt !== "string" || !prompt.trim()) {
    return response.status(400).json({ error: "Enter an instruction for the document." });
  }

  response.status(200);
  response.set({
    "content-type": "application/x-ndjson; charset=utf-8",
    "cache-control": "no-cache, no-transform",
    "x-content-type-options": "nosniff",
  });
  response.flushHeaders();
  const workerArguments = [
    "--model", modelPath,
    "--document", upload.documentPath,
    "--prompt", prompt,
  ];
  if (maxOutputTokens != null) workerArguments.push("--max-output-tokens", String(Math.max(1, Number(maxOutputTokens))));
  if (reasoningEnabled) workerArguments.push("--reasoning");
  if (reasoningEnabled && maxReasoningTokens != null) {
    workerArguments.push("--max-reasoning-tokens", String(Math.max(1, Number(maxReasoningTokens))));
  }
  const worker = spawn(inferenceLauncher, workerArguments, { stdio: ["ignore", "pipe", "pipe"] });
  worker.stdout.pipe(response, { end: false });
  worker.stderr.on("data", (chunk) => console.error(chunk.toString()));
  worker.on("error", (error) => {
    response.write(`${JSON.stringify({ type: "error", message: error.message })}\n`);
    response.end();
  });
  worker.on("close", async (exitCode) => {
    if (exitCode && !response.writableEnded) {
      response.write(`${JSON.stringify({ type: "error", message: `Inference worker exited with code ${exitCode}.` })}\n`);
    }
    response.end();
    uploads.delete(uploadId);
    await rm(upload.documentPath, { force: true });
  });
  response.on("close", () => {
    if (!response.writableEnded) worker.kill("SIGTERM");
  });
});

app.get("/api/runner", (_request, response) => response.json(runManager.snapshot()));
app.post("/api/runner/start", async (request, response, next) => {
  try { response.status(202).json(await runManager.start(request.body)); } catch (error) { next(error); }
});
app.post("/api/runner/stop", (_request, response) => response.json(runManager.stop()));
app.post("/api/benchmarks/:id/resume", async (request, response, next) => {
  try {
    if (path.basename(request.params.id) !== request.params.id) return response.status(400).json({ error: "Invalid run id." });
    const runDir = path.resolve(runsDir, request.params.id);
    if (!runDir.startsWith(`${runsDir}${path.sep}`)) return response.status(400).json({ error: "Invalid run path." });
    response.status(202).json(await runManager.resume({
      runDir,
      maxContext: request.body.maxContext,
      swapStopGib: request.body.swapStopGib,
    }));
  } catch (error) { next(error); }
});

app.get("/api/benchmarks", async (_request, response, next) => {
  try {
    const benchmarks = await listBenchmarks(runsDir);
    response.json({
      benchmarks,
      modelBenchmarks: buildModelBenchmarks(benchmarks),
      metrics: METRICS,
    });
  } catch (error) {
    next(error);
  }
});

app.get("/api/benchmarks/:id", async (request, response, next) => {
  try {
    const benchmark = await getBenchmark(runsDir, request.params.id);
    if (!benchmark) return response.status(404).json({ error: "Benchmark run was not found." });
    response.json(benchmark);
  } catch (error) {
    next(error);
  }
});

if (production) {
  const distRoot = path.join(viewerRoot, "dist");
  const indexPath = path.join(distRoot, "index.html");
  if (!existsSync(indexPath)) {
    throw new Error("Production viewer is not built. Run `npm run build` first.");
  }
  app.use(express.static(distRoot, { index: "index.html", maxAge: "1h" }));
  app.use((request, response, next) => {
    if (request.method !== "GET" || request.path.startsWith("/api/")) return next();
    response.sendFile(indexPath);
  });
} else {
  const [{ default: react }, { createServer: createViteServer }] = await Promise.all([
    import("@vitejs/plugin-react"),
    import("vite"),
  ]);
  const vite = await createViteServer({
    root: viewerRoot,
    appType: "spa",
    plugins: [react()],
    server: { middlewareMode: true, hmr: { server: httpServer } },
  });
  app.use(vite.middlewares);
}

app.use((error, _request, response, _next) => {
  console.error(error);
  response.status(500).json({ error: error.message || "The request could not be completed." });
});

let requestedPort = Number(process.env.PORT || 5173);
const host = process.env.HOST || "0.0.0.0";
httpServer.on("error", (error) => {
  if (error.code === "EADDRINUSE") {
    requestedPort += 1;
    console.warn(`Port ${requestedPort - 1} is occupied; trying ${requestedPort}.`);
    setTimeout(() => httpServer.listen(requestedPort, host), 20);
    return;
  }
  throw error;
});
httpServer.on("listening", () => {
  const address = httpServer.address();
  console.log(`\nBenchmark viewer (${production ? "production" : "development"}): http://localhost:${address.port}`);
  for (const addresses of Object.values(os.networkInterfaces())) {
    for (const item of addresses || []) {
      if (item.family === "IPv4" && !item.internal) console.log(`Network viewer:   http://${item.address}:${address.port}`);
    }
  }
  console.log(`Watching runs in: ${runsDir}\n`);
});
httpServer.listen(requestedPort, host);
