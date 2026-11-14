import { spawnSync } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import { join } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { parseArgs as parseNodeArgs } from "node:util";
import { createSheetsService } from "../lib/sheets/google-sheets-client.mjs";
import { googleSheetsSpreadsheetId } from "../lib/core/project-config.mjs";

const defaultRunsDir = "tmp/intake-runs";
const defaultDelayMs = 2000;

function printUsage() {
  console.log(`Usage:
  pnpm sheets:apply-intakes [options]

Options:
  --runs-dir <path>       Directory containing intake runs. Default: tmp/intake-runs.
  --spreadsheet-id <id>  Google Sheets spreadsheet ID. Default: config/project.json googleSheets.spreadsheetId.
  --delay-ms <ms>        Delay between intake writes in milliseconds to stay within quota. Default: 2000.
  --write                Apply changes to Google Sheets. Without this flag, only dry-runs are performed.
  --help, -h             Show this help.

This command iterates over intake run directories in <runs-dir>, automatically:
1. Queries the remote Google Sheet directly for applied batch_ids.
2. Skips runs whose batch_id has already been applied.
3. Automatically throttles requests and retries on HTTP 429 quota limits with exponential backoff.
4. If an intake run was generated against an older state of Google Sheets,
   it automatically re-reconciles against current export before applying
   (reusing cached Flickr album inventory without refetching from Flickr).
5. After each write, automatically refreshes local export (only photos & albums to save quota)
   so subsequent runs have the exact baseline state.`);
}

export function parseArgs(argv) {
  const { values } = parseNodeArgs({
    args: argv.slice(2).filter((arg) => arg !== "--"),
    options: {
      "runs-dir": { type: "string" },
      "spreadsheet-id": { type: "string" },
      "delay-ms": { type: "string" },
      write: { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });

  return {
    help: values.help ?? false,
    runsDir: values["runs-dir"] ?? defaultRunsDir,
    spreadsheetId: values["spreadsheet-id"] ?? googleSheetsSpreadsheetId,
    delayMs: Number(values["delay-ms"] ?? defaultDelayMs),
    write: values.write ?? false,
  };
}

async function fetchRemoteAppliedBatches(spreadsheetId) {
  try {
    const sheets = await createSheetsService();
    const res = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range: "import_batches!A:A",
    });
    const rows = res.data.values ?? [];
    const set = new Set();
    for (const row of rows.slice(1)) {
      if (row[0]?.trim()) {
        set.add(row[0].trim());
      }
    }
    return set;
  } catch (error) {
    console.warn(`Could not fetch remote batches: ${error.message}`);
    return new Set();
  }
}

function runCommandWithRetry(command, args, maxRetries = 4) {
  for (let attempt = 1; attempt <= maxRetries; attempt += 1) {
    const result = spawnSync(command, args, { stdio: "pipe", encoding: "utf8" });
    if (result.stdout) process.stdout.write(result.stdout);
    if (result.stderr) process.stderr.write(result.stderr);

    if (!result.error && result.status === 0) {
      return { success: true, isQuotaError: false };
    }

    const output = (result.stdout || "") + (result.stderr || "");
    const isQuota = /quota exceeded|HTTP 429/i.test(output);
    const isNetworkError = /ENOTFOUND|ETIMEDOUT|ECONNRESET|fetch failed|network timeout/i.test(output);

    if ((isQuota || isNetworkError) && attempt < maxRetries) {
      const waitSec = Math.pow(2, attempt) * 15; // 30s, 60s, 120s...
      const reason = isQuota ? "Google Sheets 429 quota" : "Transient network error";
      console.warn(`\n[Retryable Error] ${reason}. Backing off for ${waitSec}s before retry ${attempt}/${maxRetries}...`);
      spawnSync("sleep", [String(waitSec)]);
      continue;
    }

    return { success: false, isQuotaError: isQuota };
  }
  return { success: false, isQuotaError: true };
}

async function main() {
  const options = parseArgs(process.argv);
  if (options.help) {
    printUsage();
    return;
  }

  // 1. Fetch live applied batches from Google Sheets directly
  const appliedBatches = await fetchRemoteAppliedBatches(options.spreadsheetId);

  let runEntries = [];
  try {
    runEntries = await readdir(options.runsDir);
  } catch (error) {
    throw new Error(`Could not read runs directory ${options.runsDir}: ${error.message}`);
  }

  // Filter and sort intake run directories
  const intakeDirs = runEntries
    .filter((name) => name.startsWith("intake-") && !name.startsWith("intake-catalog-"))
    .sort();

  if (intakeDirs.length === 0) {
    console.log(`No intake run directories found in ${options.runsDir}.`);
    return;
  }

  console.log(`Found ${intakeDirs.length} intake run(s) in ${options.runsDir}.`);
  console.log(`Remote applied batches: ${appliedBatches.size}. Mode: ${options.write ? "WRITE (Applying to Sheets)" : "DRY-RUN (Validation only)"}.\n`);

  let appliedCount = 0;
  let skippedCount = 0;
  let errorCount = 0;

  for (const [index, dirName] of intakeDirs.entries()) {
    const runDir = join(options.runsDir, dirName);
    const batchCsvPath = join(runDir, "import-batch.csv");
    const summaryJsonPath = join(runDir, "summary.json");

    let batchId = "";
    let albumId = "";
    try {
      const summaryContent = await readFile(summaryJsonPath, "utf8");
      const summary = JSON.parse(summaryContent);
      albumId = summary.album_id ?? "";
      const batchContent = await readFile(batchCsvPath, "utf8");
      batchId = batchContent.split("\n")[1]?.split(",")[0]?.trim() ?? "";
    } catch {
      console.warn(`[Skip] [${index + 1}/${intakeDirs.length}] ${dirName}: summary.json or import-batch.csv not found.`);
      skippedCount += 1;
      continue;
    }

    if (batchId && appliedBatches.has(batchId)) {
      console.log(`[Skip] [${index + 1}/${intakeDirs.length}] ${dirName} (batch ${batchId} already applied)`);
      skippedCount += 1;
      continue;
    }

    console.log(`\n=== [${index + 1}/${intakeDirs.length}] ${options.write ? "Applying" : "Checking"} ${dirName} (Album ${albumId}) ===`);

    let applyArgs = ["scripts/commands/apply-intake-run.mjs", "--run-dir", runDir];
    if (options.write) {
      applyArgs.push("--write");
    }

    let { success, isQuotaError } = runCommandWithRetry(process.execPath, applyArgs);

    // If failed and not purely an unrecoverable quota error, attempt re-reconciliation if state was out of sync
    if (!success && albumId) {
      console.log(`\nNotice: run state out of sync or failed. Re-reconciling album ${albumId} with cached inventory...`);
      const reconcile = runCommandWithRetry(process.execPath, [
        "scripts/commands/run-intake.mjs",
        "--album",
        albumId,
        "--run-id",
        dirName,
      ]);

      if (reconcile.success) {
        console.log(`Re-reconciliation succeeded. Retrying apply for ${dirName}...`);
        const retryResult = runCommandWithRetry(process.execPath, applyArgs);
        success = retryResult.success;
      }
    }

    if (!success) {
      console.error(`Error processing intake run ${dirName}. Stopping batch.`);
      errorCount += 1;
      process.exitCode = 1;
      break;
    }

    // Refresh local export if we wrote to Sheets (export only necessary sheets: photos, albums, import_batches)
    if (options.write) {
      console.log("Exporting updated Sheets state (photos, albums, import_batches)...");
      runCommandWithRetry(process.execPath, [
        "scripts/commands/export-sheets.mjs",
        "--sheets",
        "photos,albums,import_batches",
      ]);
      if (batchId) {
        appliedBatches.add(batchId);
      }
      if (options.delayMs > 0) {
        await sleep(options.delayMs);
      }
    }

    appliedCount += 1;
  }

  console.log(`\nSummary:`);
  console.log(`- Processed / Applied: ${appliedCount}`);
  console.log(`- Skipped: ${skippedCount}`);
  if (errorCount > 0) {
    console.log(`- Errors: ${errorCount}`);
  }
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/^.*\//, ""))) {
  main().catch((error) => {
    console.error(`Could not apply intake runs: ${error.message}`);
    process.exitCode = 1;
  });
}
