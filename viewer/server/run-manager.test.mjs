import assert from "node:assert/strict";
import test from "node:test";

import { buildContextCaps, buildDecodeVariants } from "./run-manager.mjs";

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
