async function request(path) {
  const response = await fetch(path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.error || `Request failed with status ${response.status}.`);
  }
  return response.json();
}

export const fetchBenchmarks = () => request("/api/benchmarks");
export const fetchBenchmark = (id) => request(`/api/benchmarks/${encodeURIComponent(id)}`);
export const fetchModels = () => request("/api/models");
export const fetchRunner = () => request("/api/runner");

export function uploadInferenceDocument(file, onProgress) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", "/api/inference/files");
    request.setRequestHeader("content-type", "text/plain; charset=utf-8");
    request.setRequestHeader("x-file-name", encodeURIComponent(file.name));
    request.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(event.loaded / event.total);
    };
    request.onerror = () => reject(new Error("Document upload failed."));
    request.onload = () => {
      const payload = JSON.parse(request.responseText || "{}");
      if (request.status >= 200 && request.status < 300) resolve(payload);
      else reject(new Error(payload.error || `Upload failed with status ${request.status}.`));
    };
    request.send(file);
  });
}

export async function streamInference(options, onEvent, signal) {
  const response = await fetch("/api/inference/generate", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(options),
    signal,
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `Inference failed with status ${response.status}.`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";
  while (true) {
    const { done, value } = await reader.read();
    pending += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = pending.split("\n");
    pending = lines.pop() || "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (pending.trim()) onEvent(JSON.parse(pending));
}

async function post(path, body = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.error || `Request failed with status ${response.status}.`);
  return payload;
}

export const startRunner = (options) => post("/api/runner/start", options);
export const stopRunner = () => post("/api/runner/stop");
export const resumeBenchmark = (id, maxContext, swapStopGib) => post(`/api/benchmarks/${encodeURIComponent(id)}/resume`, { maxContext, swapStopGib });
