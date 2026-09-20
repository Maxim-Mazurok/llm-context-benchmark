import assert from "node:assert/strict";
import test from "node:test";

import { createServerSentEventParser, discoverUnslothEndpoint, normalizeUnslothEndpoint } from "./unsloth-provider.mjs";

// cspell:words Unsloth unsloth

test("Unsloth endpoints are normalized and restricted to loopback", () => {
  assert.equal(normalizeUnslothEndpoint("http://localhost:8888"), "http://localhost:8888/v1");
  assert.equal(normalizeUnslothEndpoint("http://127.0.0.1:9000/v1/"), "http://127.0.0.1:9000/v1");
  assert.throws(() => normalizeUnslothEndpoint("https://example.com/v1"), /local HTTP endpoint/);
});

test("fragmented server-sent events are decoded without completion sentinels", () => {
  const events = [];
  const parser = createServerSentEventParser((event) => events.push(event));
  parser.push("data: {\"choices\":[{\"delta\":");
  parser.push("{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n");
  parser.push("\n");
  parser.finish();
  assert.deepEqual(events, [{ choices: [{ delta: { content: "hello" } }] }]);
});

test("Unsloth discovery skips unrelated services and finds an authenticated API", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url) => {
    if (String(url).includes(":8889/")) {
      return new Response(JSON.stringify({ error: { message: "Not authenticated" } }), {
        status: 401,
        headers: { "content-type": "application/json" },
      });
    }
    return new Response("Proxy authentication required", { status: 407, headers: { "content-type": "text/html" } });
  };
  assert.equal(await discoverUnslothEndpoint(), "http://127.0.0.1:8889/v1");
});