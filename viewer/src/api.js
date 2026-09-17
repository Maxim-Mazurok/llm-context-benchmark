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
export const resumeBenchmark = (id, maxContext) => post(`/api/benchmarks/${encodeURIComponent(id)}/resume`, { maxContext });
