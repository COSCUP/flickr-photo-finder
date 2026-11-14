import { spawnSync } from "node:child_process";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { basename, join } from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs as parseNodeArgs } from "node:util";
import {
  readAlbumCatalog,
  resolveAlbumInput,
} from "../lib/flickr/album-catalog.mjs";
import { parseCsv } from "../lib/core/csv-utils.mjs";
import { importBatchHeaders } from "../lib/core/photo-schema.mjs";
import { sheetsExportAlbumsPath, sheetsExportPhotosPath } from "../lib/core/workflow-paths.mjs";

const defaultRunsDir = "tmp/intake-runs";

function printUsage() {
  console.log(`Usage:
  pnpm intake:run -- (--album <album-id> | --all-albums | --resume <run-dir>) [options]

Options:
  --album <value>         Album ID from the albums CSV. Flickr album URLs are accepted for debugging. ('all' can be used as alias for --all-albums).
  --all-albums            Build a complete ordered membership baseline for every catalog album (or all unprocessed albums when combined with --unprocessed).
  --unprocessed           When combined with --all-albums, iterate through and process only albums where last_processed_at is empty.
  --resume <path>         Resume an interrupted intake run directory, skipping already completed steps.
  --albums <path>         Google Sheets albums CSV export or local fixture. Default: tmp/sheets-export/albums.csv.
  --photos-export <path>  Current Google Sheets photos CSV export for duplicate detection. Default: tmp/sheets-export/photos.csv.
  --input <html-file>     Read saved Flickr album HTML instead of fetching the album page.
  --runs-dir <path>       Directory for intake run artifacts. Default: tmp/intake-runs.
  --run-id <value>        Explicit run ID. Default: intake-<album-id>-<timestamp>.
  --imported-at <value>   Import timestamp to write. Default: current time as ISO string.
  --operator <value>      Operator or agent name to record in import_batches.
  --no-validate           Skip validation for generated CSV outputs.

Each run writes:
  photos-to-append.csv
  albums-updated.csv
  import-batch.csv
  reconciliation.json
  inventories.json
  summary.json`);
}

export function parseArgs(argv) {
  const { values } = parseNodeArgs({
    args: argv.slice(2).filter((arg) => arg !== "--"),
    options: {
      album: { type: "string" },
      "all-albums": { type: "boolean" },
      unprocessed: { type: "boolean" },
      resume: { type: "string" },
      albums: { type: "string" },
      "photos-export": { type: "string" },
      input: { type: "string" },
      "runs-dir": { type: "string" },
      "run-id": { type: "string" },
      "imported-at": { type: "string" },
      operator: { type: "string" },
      "no-validate": { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });
  let album = values.album ?? "";
  let allAlbums = values["all-albums"] ?? false;
  if (album.toLowerCase() === "all") {
    album = "";
    allAlbums = true;
  }
  const options = {
    album,
    allAlbums,
    unprocessed: values.unprocessed ?? false,
    resume: values.resume ?? "",
    albums: values.albums ?? sheetsExportAlbumsPath,
    help: values.help ?? false,
    importedAt: values["imported-at"] ?? "",
    input: values.input ?? "",
    operator: values.operator ?? "",
    photosExport: values["photos-export"] ?? sheetsExportPhotosPath,
    runId: values["run-id"] ?? "",
    runsDir: values["runs-dir"] ?? defaultRunsDir,
    validate: !(values["no-validate"] ?? false),
  };

  if (!options.help) {
    const scopeCount = Number(Boolean(options.album)) + Number(options.allAlbums) + Number(Boolean(options.resume));
    if (scopeCount !== 1) {
      throw new Error("Pass exactly one of --album, --all-albums, or --resume");
    }
    if (options.unprocessed && !options.allAlbums) {
      throw new Error("--unprocessed can only be used with --all-albums (or --album all)");
    }
    if (!options.albums) {
      throw new Error("--albums requires a path");
    }
    if (!options.photosExport) {
      throw new Error("--photos-export requires a path");
    }
    if (!options.runsDir) {
      throw new Error("--runs-dir requires a path");
    }
    if (options.importedAt && Number.isNaN(Date.parse(options.importedAt))) {
      throw new Error("--imported-at must be a valid date or datetime");
    }
  }

  return options;
}

function makeRunId(albumId, createdAt) {
  const compactTimestamp = createdAt.replace(/\D/g, "").slice(0, 14);
  return `intake-${albumId}-${compactTimestamp}`;
}

function getCell(record, header) {
  const index = importBatchHeaders.indexOf(header);
  return index >= 0 ? record[index] ?? "" : "";
}

async function readImportBatches(path) {
  const csv = await readFile(path, "utf8");
  const [headers, ...rows] = parseCsv(csv);
  if (!headers) {
    throw new Error(`${path} is empty`);
  }
  if (headers.join(",") !== importBatchHeaders.join(",")) {
    throw new Error(`${path} headers do not match import_batches schema`);
  }
  return rows;
}

function runPhotoImport(args) {
  console.error("Progress: running photo import helper.");
  const result = spawnSync(process.execPath, ["scripts/commands/import-album-photos.mjs", ...args], {
    stdio: "inherit",
  });

  if (result.error) {
    throw result.error;
  }

  if (result.status !== 0) {
    throw new Error("photo import run failed");
  }
}

async function main() {
  const options = parseArgs(process.argv);
  if (options.help) {
    printUsage();
    return;
  }

  console.error(`Progress: reading album catalog from ${options.albums}.`);
  const albums = await readAlbumCatalog(options.albums);

  if (options.unprocessed) {
    const unprocessedAlbums = albums.filter((item) => !item.last_processed_at?.trim());
    console.error(`Progress: found ${unprocessedAlbums.length} unprocessed album(s) out of ${albums.length} total in ${options.albums}.`);
    if (unprocessedAlbums.length === 0) {
      console.log("No unprocessed albums found.");
      return;
    }

    // Inspect existing runs in options.runsDir
    let existingRunDirs = [];
    try {
      existingRunDirs = await readdir(options.runsDir);
    } catch {}

    const completedAlbums = new Set();
    const latestRunDirByAlbumId = new Map();
    for (const dirName of existingRunDirs) {
      const match = dirName.match(/^intake-(\d+)-/);
      if (match) {
        const aid = match[1];
        const dirPath = join(options.runsDir, dirName);
        latestRunDirByAlbumId.set(aid, dirPath);
        try {
          const summary = JSON.parse(await readFile(join(dirPath, "summary.json"), "utf8"));
          if (summary.run_id && summary.album_id === aid) {
            completedAlbums.add(aid);
          }
        } catch {}
      }
    }

    const pendingAlbums = unprocessedAlbums.filter((album) => !completedAlbums.has(album.album_id));
    console.error(`Progress: ${completedAlbums.size} album(s) already have completed intake runs. ${pendingAlbums.length} remaining to process.`);

    const runs = [];
    for (const [index, unprocAlbum] of pendingAlbums.entries()) {
      const aid = unprocAlbum.album_id;
      const existingPartialDir = latestRunDirByAlbumId.get(aid);
      const isResuming = Boolean(existingPartialDir);

      console.error(
        `\n=== [${index + 1}/${pendingAlbums.length}] ${isResuming ? "Resuming" : "Processing"} album ${aid} (${unprocAlbum.album_title || "Untitled"}) ===`,
      );

      let singleArgs = [];
      let albumRunId = "";
      if (isResuming) {
        albumRunId = basename(existingPartialDir);
        singleArgs = [
          "--resume",
          existingPartialDir,
          "--albums",
          options.albums,
          "--photos-export",
          options.photosExport,
        ];
      } else {
        albumRunId = makeRunId(aid, new Date().toISOString());
        singleArgs = [
          "--album",
          aid,
          "--albums",
          options.albums,
          "--photos-export",
          options.photosExport,
          "--runs-dir",
          options.runsDir,
          "--run-id",
          albumRunId,
        ];
      }

      if (options.operator) {
        singleArgs.push("--operator", options.operator);
      }
      if (!options.validate) {
        singleArgs.push("--no-validate");
      }

      const result = spawnSync(process.execPath, [process.argv[1], ...singleArgs], {
        stdio: "inherit",
      });
      if (result.error) {
        throw result.error;
      }
      if (result.status !== 0) {
        throw new Error(`intake failed for album ${aid}`);
      }
      runs.push({ albumId: aid, runId: albumRunId });
    }

    console.log(`\nSuccessfully processed ${runs.length} album(s).`);
    return;
  }

  let runDir = "";
  let runId = "";
  let createdAt = options.importedAt || new Date().toISOString();
  let albumId = "";
  let albumUrl = "";
  let album = {};
  let isAllAlbums = options.allAlbums;

  if (options.resume) {
    runDir = options.resume;
    runId = basename(runDir);
    console.error(`Progress: resuming intake run from ${runDir}.`);
    // Check if reconciliation.json already exists to recover scope and album info
    const reconciliationPath = join(runDir, "reconciliation.json");
    try {
      const rec = JSON.parse(await readFile(reconciliationPath, "utf8"));
      isAllAlbums = rec.scope === "catalog";
      if (!isAllAlbums && rec.album_order?.length > 0) {
        albumId = rec.album_order[0];
      }
    } catch {
      // If reconciliation.json doesn't exist yet, try to inspect runId
      if (runId.startsWith("intake-catalog-")) {
        isAllAlbums = true;
      }
    }
  } else {
    console.error(`Progress: resolving ${options.allAlbums ? "complete album catalog" : `album ${options.album}`}.`);
    const resolved = options.allAlbums
      ? { albumId: "catalog", albumUrl: "" }
      : await resolveAlbumInput(options.album, options.albums);
    albumId = resolved.albumId;
    albumUrl = resolved.albumUrl;
    runId = options.runId || makeRunId(albumId, createdAt);
    runDir = join(options.runsDir, runId);
    console.error(`Progress: creating intake run directory ${runDir}.`);
    await mkdir(options.runsDir, { recursive: true });
    await mkdir(runDir, { recursive: true });
  }

  if (albumId && albumId !== "catalog") {
    album = albums.find((item) => item.album_id === albumId) ?? {};
    if (!albumUrl) {
      albumUrl = album.album_url ?? "";
    }
  }

  const photosOutput = join(runDir, "photos-to-append.csv");
  const albumsOutput = join(runDir, "albums-updated.csv");
  const batchOutput = join(runDir, "import-batch.csv");
  const reconciliationOutput = join(runDir, "reconciliation.json");
  const inventoriesOutput = join(runDir, "inventories.json");
  const summaryOutput = join(runDir, "summary.json");

  const photoImportArgs = [
    ...(isAllAlbums ? ["--all-albums"] : ["--album", albumId]),
    "--albums",
    options.albums,
    "--photos-export",
    options.photosExport,
    "--output",
    photosOutput,
    "--albums-output",
    albumsOutput,
    "--batch-output",
    batchOutput,
    "--reconciliation-output",
    reconciliationOutput,
    "--imported-at",
    createdAt,
    "--source-tool",
    "pnpm intake:run",
  ];

  // If inventories.json already exists in runDir, use it to skip fetching album photo lists!
  try {
    await readFile(inventoriesOutput, "utf8");
    console.error(`Progress: reusing existing album inventories from ${inventoriesOutput}.`);
    photoImportArgs.push("--inventories-input", inventoriesOutput);
  } catch {
    photoImportArgs.push("--inventories-output", inventoriesOutput);
  }

  if (options.input) {
    photoImportArgs.push("--input", options.input);
  }
  if (options.operator) {
    photoImportArgs.push("--operator", options.operator);
  }
  if (!options.validate) {
    photoImportArgs.push("--no-validate");
  }

  runPhotoImport(photoImportArgs);

  console.error(`Progress: reading generated reconciliation artifacts.`);
  const [batches, reconciliation] = await Promise.all([
    readImportBatches(batchOutput),
    readFile(reconciliationOutput, "utf8").then(JSON.parse),
  ]);
  const batch = batches[0] ?? [];
  const foundPhotoCount = isAllAlbums
    ? reconciliation.desired_photo_ids.length
    : Number(getCell(batch, "found_photo_count"));
  const summary = {
    run_id: runId,
    created_at: createdAt,
    scope: reconciliation.scope,
    album_id: isAllAlbums ? "" : albumId,
    album_url: albumUrl,
    album_title: album.album_title ?? "",
    operator: options.operator,
    source_tool: "pnpm intake:run",
    found_photo_count: foundPhotoCount,
    new_photo_count: reconciliation.counts.new,
    skipped_photo_count: foundPhotoCount - reconciliation.counts.new,
    membership_update_count: reconciliation.counts.membership_updated,
    deleted_photo_count: reconciliation.counts.deleted,
    reordered_photo_count: reconciliation.counts.reordered,
    outputs: {
      photos_to_append: photosOutput,
      albums_updated: albumsOutput,
      import_batch: batchOutput,
      reconciliation: reconciliationOutput,
      inventories: inventoriesOutput,
      summary: summaryOutput,
    },
  };

  console.error(`Progress: writing intake run summary to ${summaryOutput}.`);
  await writeFile(summaryOutput, `${JSON.stringify(summary, null, 2)}\n`);
  console.log(`Wrote intake run summary to ${summaryOutput}.`);
  console.log(`Intake run directory: ${runDir}`);
  console.log(`Intake run ${runId} is ready for review before applying to Google Sheets.`);
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  try {
    await main();
  } catch (error) {
    console.error(`Could not run album intake: ${error.message}`);
    process.exitCode = 1;
  }
}
