import assert from "node:assert/strict";
import test from "node:test";

import { buildBenchmarkArguments, buildContextCaps, buildDecodeVariants } from "./run-manager.mjs";

test("an empty final target adds an uncapped final stage", () => {
  assert.deepEqual(buildContextCaps("staged", null, [32768, 8192, 16384]), [8192, 16384, 32768, null]);
  assert.deepEqual(buildContextCaps("continuous", null, []), [null]);
});

test("a final target caps and completes staged runs", () => {
  assert.deepEqual(buildContextCaps("staged", 20000, [8192, 16384, 32768]), [8192, 16384, 20000]);
});

test("raw plus speculative creates both variants only when supported", () => {
  assert.deepEqual(buildDecodeVariants({ speculative: [{}] }, "both"), [false, true]);
  assert.deepEqual(buildDecodeVariants({ speculative: [] }, "both"), [false]);
  assert.deepEqual(buildDecodeVariants({ speculative: [] }, "speculative"), []);
});

test("resume arguments include final context and maximum added swap", () => {
  assert.deepEqual(
    buildBenchmarkArguments({ resume: true, output: "/runs/example", cap: 131072, swapStopGib: 8 }),
    ["--resume", "/runs/example", "--max-context", "131072", "--swap-stop-gib", "8"],
  );
  assert.deepEqual(
    buildBenchmarkArguments({ resume: true, output: "/runs/example", cap: null, swapStopGib: 4 }),
    ["--resume", "/runs/example", "--no-max-context", "--swap-stop-gib", "4"],
  );
});
