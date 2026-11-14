import assert from "node:assert/strict";
import test from "node:test";
import { parseArgs } from "../scripts/commands/run-intake.mjs";
import { parseArgs as parseValidationArgs } from "../scripts/commands/validate-intake-run.mjs";

test("intake run arguments", async (t) => {
  await t.test("accepts the pnpm separator before the catalog baseline flag", () => {
    const options = parseArgs(["node", "run-intake.mjs", "--", "--all-albums"]);

    assert.equal(options.allAlbums, true);
    assert.equal(options.album, "");
  });

  await t.test("requires exactly one reconciliation scope", () => {
    assert.throws(
      () => parseArgs(["node", "run-intake.mjs", "--album", "123", "--all-albums"]),
      /exactly one/,
    );
  });

  await t.test("accepts resume run directory as scope", () => {
    const options = parseArgs(["node", "run-intake.mjs", "--", "--resume", "tmp/intake-runs/run-1"]);

    assert.equal(options.resume, "tmp/intake-runs/run-1");
    assert.equal(options.allAlbums, false);
    assert.equal(options.album, "");
  });

  await t.test("accepts --unprocessed with --all-albums", () => {
    const options = parseArgs(["node", "run-intake.mjs", "--all-albums", "--unprocessed"]);

    assert.equal(options.allAlbums, true);
    assert.equal(options.unprocessed, true);
    assert.equal(options.album, "");
  });

  await t.test("normalizes --album all to --all-albums", () => {
    const options = parseArgs(["node", "run-intake.mjs", "--album", "all", "--unprocessed"]);

    assert.equal(options.allAlbums, true);
    assert.equal(options.unprocessed, true);
    assert.equal(options.album, "");
  });

  await t.test("rejects --unprocessed with single album ID", () => {
    assert.throws(
      () => parseArgs(["node", "run-intake.mjs", "--album", "72177720335626334", "--unprocessed"]),
      /--unprocessed can only be used with --all-albums/,
    );
  });
});

test("intake validation accepts the pnpm separator", () => {
  const options = parseValidationArgs(["node", "validate-intake-run.mjs", "--", "--run-dir", "tmp/run"]);

  assert.equal(options.runDir, "tmp/run");
});
