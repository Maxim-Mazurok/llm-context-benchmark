import { spawn } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
import os from "node:os";
import path from "node:path";

// cspell:words Unsloth unsloth

const DEFAULT_ENDPOINT = "http://127.0.0.1:8888/v1";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);

export function normalizeUnslothEndpoint(value = DEFAULT_ENDPOINT) {
  const endpoint = new URL(value);
  if (endpoint.protocol !== "http:" || !LOOPBACK_HOSTS.has(endpoint.hostname)) {
    throw new Error("Unsloth Studio must use a local HTTP endpoint.");
  }
  endpoint.username = "";
  endpoint.password = "";
  endpoint.search = "";
  endpoint.hash = "";
  endpoint.pathname = `${endpoint.pathname.replace(/\/+$/, "").replace(/\/v1$/, "")}/v1`;
  return endpoint.toString().replace(/\/$/, "");
}

export function createServerSentEventParser(onEvent) {
  let pending = "";
  return {
    push(value) {
      pending += value.replaceAll("\r\n", "\n");
      const frames = pending.split("\n\n");
      pending = frames.pop() || "";
      for (const frame of frames) {
        const data = frame.split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trimStart())
          .join("\n");
        if (data && data !== "[DONE]") onEvent(JSON.parse(data));
      }
    },
    finish() {
      if (pending.trim()) this.push("\n\n");
    },
  };
}

function authorizationHeaders(apiKey) {
  return apiKey ? { authorization: `Bearer ${apiKey}` } : {};
}

export async function listUnslothModels({ endpoint, apiKey }) {
  const baseEndpoint = normalizeUnslothEndpoint(endpoint);
  let response;
  try {
    response = await fetch(`${baseEndpoint}/models`, {
      headers: authorizationHeaders(apiKey),
      signal: AbortSignal.timeout(5_000),
    });
  } catch {
    throw new Error("Cannot connect to Unsloth Studio. Start it, then refresh models.");
  }
  if (!response.ok) {
    throw new Error(response.status === 401
      ? "Unsloth rejected the API key. Create one in Settings > API."
      : `Unsloth model discovery failed with status ${response.status}.`);
  }
  const payload = await response.json();
  return (payload.data || []).map((model) => ({ id: model.id, name: model.id }));
}

export async function probeUnslothStudio(endpoint = DEFAULT_ENDPOINT) {
  const baseEndpoint = normalizeUnslothEndpoint(endpoint);
  try {
    const response = await fetch(`${baseEndpoint}/models`, { signal: AbortSignal.timeout(1_000) });
    const contentType = response.headers.get("content-type") || "";
    return contentType.includes("application/json") && [200, 401, 403].includes(response.status);
  } catch {
    return false;
  }
}

export async function discoverUnslothEndpoint(preferredEndpoint = DEFAULT_ENDPOINT) {
  const normalizedPreferredEndpoint = normalizeUnslothEndpoint(preferredEndpoint);
  const candidateEndpoints = [
    normalizedPreferredEndpoint,
    ...Array.from({ length: 12 }, (_, index) => `http://127.0.0.1:${8888 + index}/v1`),
  ];
  for (const endpoint of new Set(candidateEndpoints)) {
    if (await probeUnslothStudio(endpoint)) return endpoint;
  }
  return normalizedPreferredEndpoint;
}

export async function streamUnslothCompletion(options, onEvent) {
  const baseEndpoint = normalizeUnslothEndpoint(options.endpoint);
  const response = await fetch(`${baseEndpoint}/chat/completions`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      ...authorizationHeaders(options.apiKey),
    },
    body: JSON.stringify(options.requestBody),
    signal: options.signal,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error?.message || payload.error || `Unsloth inference failed with status ${response.status}.`);
  }
  if (!response.body) throw new Error("Unsloth returned an empty response stream.");
  const decoder = new TextDecoder();
  const parser = createServerSentEventParser(onEvent);
  for await (const chunk of response.body) parser.push(decoder.decode(chunk, { stream: true }));
  parser.push(decoder.decode());
  parser.finish();
}

export function unslothInstallation() {
  const commandPath = path.join(os.homedir(), ".local", "bin", "unsloth");
  const applicationPath = path.join(os.homedir(), "Applications", "Unsloth Studio.app");
  const serverPath = path.join(os.homedir(), ".unsloth", "llama.cpp", "build", "bin", "llama-server");
  return {
    installed: existsSync(commandPath) || existsSync(applicationPath),
    commandPath: existsSync(commandPath) ? commandPath : "unsloth",
    serverPath,
    benchmarkInstalled: existsSync(serverPath),
  };
}

function directoryEntries(directory) {
  try {
    return readdirSync(directory, { withFileTypes: true });
  } catch {
    return [];
  }
}

export function discoverUnslothBenchmarkModels() {
  const hubDirectory = path.join(os.homedir(), ".cache", "huggingface", "hub");
  const models = [];
  for (const repository of directoryEntries(hubDirectory)) {
    if (!repository.isDirectory() || !repository.name.startsWith("models--")) continue;
    const repositoryName = repository.name.slice("models--".length).replaceAll("--", "/");
    const snapshotsDirectory = path.join(hubDirectory, repository.name, "snapshots");
    for (const revision of directoryEntries(snapshotsDirectory)) {
      if (!revision.isDirectory()) continue;
      const revisionDirectory = path.join(snapshotsDirectory, revision.name);
      for (const file of directoryEntries(revisionDirectory)) {
        const lowerName = file.name.toLowerCase();
        if (!lowerName.endsWith(".gguf") || lowerName.startsWith("mmproj-")) continue;
        const shard = lowerName.match(/-(\d{5})-of-\d{5}\.gguf$/);
        if (shard && shard[1] !== "00001") continue;
        const modelPath = path.join(revisionDirectory, file.name);
        models.push({
          id: modelPath,
          path: modelPath,
          relativeName: `${repositoryName}/${file.name}`,
          speculative: [],
        });
      }
    }
  }
  return models.sort((left, right) => left.relativeName.localeCompare(right.relativeName));
}

export function startUnslothStudio(endpoint = DEFAULT_ENDPOINT) {
  const baseEndpoint = new URL(normalizeUnslothEndpoint(endpoint));
  const installation = unslothInstallation();
  if (!installation.installed) throw new Error("Unsloth Studio is not installed.");
  const child = spawn(installation.commandPath, [
    "studio", "--host", "127.0.0.1", "--port", baseEndpoint.port || "80",
  ], { detached: true, stdio: "ignore" });
  child.unref();
  return { endpoint: baseEndpoint.toString().replace(/\/v1$/, ""), processId: child.pid };
}

export const defaultUnslothEndpoint = DEFAULT_ENDPOINT;