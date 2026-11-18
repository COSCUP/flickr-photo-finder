import { mkdir, readFile, rename, rm, stat, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs as parseNodeArgs } from "node:util";
import { createProgressThrottle } from "../lib/core/progress.mjs";
import { googleSheetsSpreadsheetId } from "../lib/core/project-config.mjs";
import {
  createEmbeddingClient,
  defaultEmbeddingEngine,
  embeddingApiKeyEnv,
  embeddingEngineDefaults,
  embeddingEngines,
  jpegDataUrl,
  mlxServeImageConcurrency,
} from "../lib/finder/embedding-client.mjs";
import {
  buildEmbeddingSegment,
  compactEntries,
  computeEmbeddingDiff,
  defaultEmbeddingsDir,
  embeddingKinds,
  embeddingRowBytes,
  mergeEmbeddingSegments,
  quantizeEmbedding,
  readEmbeddingStore,
  writeEmbeddingStore,
} from "../lib/finder/embeddings.mjs";
import { finderDataSources } from "../lib/pages/static-finder-data.mjs";
import { googleSheetsCsvUrl, readStaticFinderPayloads } from "./build-pages.mjs";

export const defaultWorkDir = "tmp/finder-embeddings";
const defaultBatchSize = 16;
const checkpointEvery = 500;
const thumbnailConcurrency = 6;
const thumbnailAttempts = 3;
const thumbnailTimeoutMs = 30000;
const kindOptions = new Set(["text", "image", "all"]);

function printUsage() {
  console.log(`Usage:
  pnpm finder:embed [options]

Options:
  --engine <engine>       mlx-serve or omlx. Default: ${defaultEmbeddingEngine}.
  --url <url>             Embedding server base URL. Default: ${embeddingEngineDefaults["mlx-serve"].url} (mlx-serve),
                          ${embeddingEngineDefaults.omlx.url} (omlx).
  --model <id>            Model id on the server. Default: ${embeddingEngineDefaults["mlx-serve"].model} (mlx-serve),
                          ${embeddingEngineDefaults.omlx.model} (omlx).
  --batch <count>         Items per batch. Default: ${defaultBatchSize}. mlx-serve sends text in one
                          request per batch but images one per request (${mlxServeImageConcurrency} in flight).
  --data-source <source>  public-csv or export. Default: public-csv.
  --spreadsheet-id <id>   Google Sheets spreadsheet ID for public-csv. Default: config/project.json.
  --kind <kind>           text, image, or all. Default: all.
  --embeddings-dir <path> Segment store directory. Default: ${defaultEmbeddingsDir}.
  --compact               Merge segments into one base segment per kind and drop photos
                          no longer in the photo index.
  --rebuild               Recompute every vector and replace existing segments of the kind
                          (use after changing the model or dimensions).
  --limit <count>         Only consider the first N photos (for testing).
  --dry-run               Print pending counts only; does not contact the embedding server.
  --help, -h              Show this help.

Environment:
  ${embeddingApiKeyEnv}    Optional bearer API key for the embedding server.

Reads photos the same way as pnpm finder:build (no Google credentials needed for
public-csv). Image vectors use Flickr _n.jpg thumbnails cached under
${defaultWorkDir}/thumbs/. Only photos without an image vector, or whose document
text sha256 changed, are embedded. Progress is checkpointed every ${checkpointEvery}
photos under ${defaultWorkDir}/ so an interrupted run resumes. Finished vectors are
appended as new segments under the embeddings directory; nothing is written when
there is no difference.`);
}

function positiveInteger(value, flag) {
  const number = Number(value);
  if (!Number.isInteger(number) || number < 1) {
    throw new Error(`${flag} must be a positive integer`);
  }
  return number;
}

function parseArgs(argv) {
  const { values } = parseNodeArgs({
    args: argv.slice(2).filter((arg) => arg !== "--"),
    options: {
      engine: { type: "string" },
      url: { type: "string" },
      model: { type: "string" },
      batch: { type: "string" },
      "data-source": { type: "string" },
      "spreadsheet-id": { type: "string" },
      kind: { type: "string" },
      "embeddings-dir": { type: "string" },
      compact: { type: "boolean" },
      rebuild: { type: "boolean" },
      limit: { type: "string" },
      "dry-run": { type: "boolean" },
      help: { type: "boolean", short: "h" },
    },
  });
  const options = {
    help: values.help ?? false,
    engine: values.engine ?? defaultEmbeddingEngine,
    url: values.url ?? "",
    model: values.model ?? "",
    batch: values.batch === undefined ? defaultBatchSize : values.batch,
    dataSource: values["data-source"] ?? "public-csv",
    spreadsheetId: values["spreadsheet-id"] ?? googleSheetsSpreadsheetId,
    kind: values.kind ?? "all",
    embeddingsDir: values["embeddings-dir"] ?? defaultEmbeddingsDir,
    compact: values.compact ?? false,
    rebuild: values.rebuild ?? false,
    limit: values.limit,
    dryRun: values["dry-run"] ?? false,
  };

  if (!options.help) {
    if (!embeddingEngines.has(options.engine)) {
      throw new Error(`--engine must be one of: ${[...embeddingEngines].join(", ")}`);
    }
    if (!finderDataSources.has(options.dataSource)) {
      throw new Error(`--data-source must be one of: ${[...finderDataSources].join(", ")}`);
    }
    if (!kindOptions.has(options.kind)) {
      throw new Error(`--kind must be one of: ${[...kindOptions].join(", ")}`);
    }
    if (!options.embeddingsDir) {
      throw new Error("--embeddings-dir requires a path");
    }
    options.batch = positiveInteger(options.batch, "--batch");
    options.limit = options.limit === undefined ? 0 : positiveInteger(options.limit, "--limit");
    if (options.compact && options.rebuild) {
      throw new Error("--compact and --rebuild cannot be combined; --rebuild already replaces segments");
    }
    if (options.limit && (options.compact || options.rebuild) && !options.dryRun) {
      throw new Error("--limit cannot be combined with --compact or --rebuild because photos outside the limit would be dropped");
    }
    if (options.dataSource === "public-csv" && !options.spreadsheetId) {
      throw new Error("Set googleSheets.spreadsheetId in config/project.json or pass --spreadsheet-id");
    }
  }

  return options;
}

async function readPhotos({ dataSource, spreadsheetId, limit }) {
  const payloads = await readStaticFinderPayloads({
    albumsCsvUrl: "",
    dataSource,
    photosCsvUrl: dataSource === "public-csv" ? googleSheetsCsvUrl(spreadsheetId, "photos") : "",
  });
  return limit ? payloads.photos.slice(0, limit) : payloads.photos;
}

function selectedKinds(kind) {
  return kind === "all" ? embeddingKinds : [kind];
}

async function fileExists(path) {
  try {
    return (await stat(path)).isFile();
  } catch (error) {
    if (error.code === "ENOENT") {
      return false;
    }
    throw error;
  }
}

function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function mapWithConcurrency(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await worker(items[index], index);
    }
  });
  await Promise.all(runners);
  return results;
}

// Returns the cached thumbnail bytes, or null when the photo cannot be downloaded.
async function loadThumbnail({ photoId, url }, thumbsDir) {
  if (!/^[A-Za-z0-9_-]+$/.test(photoId)) {
    return { photoId, error: "photo_id is not safe as a file name" };
  }
  const path = join(thumbsDir, `${photoId}.jpg`);
  if (await fileExists(path)) {
    return { photoId, data: await readFile(path) };
  }
  let lastError = "";
  for (let attempt = 1; attempt <= thumbnailAttempts; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(thumbnailTimeoutMs) });
      if (response.status === 404 || response.status === 410) {
        return { photoId, error: `HTTP ${response.status}` };
      }
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const data = Buffer.from(await response.arrayBuffer());
      if (data.length === 0) {
        throw new Error("empty response");
      }
      const partialPath = `${path}.partial`;
      await writeFile(partialPath, data);
      await rename(partialPath, path);
      return { photoId, data };
    } catch (error) {
      lastError = error.message;
      if (attempt < thumbnailAttempts) {
        await delay(1000 * 2 ** (attempt - 1));
      }
    }
  }
  return { photoId, error: lastError };
}

function checkpointPaths(workDir, kind) {
  return {
    bin: join(workDir, `partial-${kind}.bin`),
    json: join(workDir, `partial-${kind}.json`),
  };
}

async function readCheckpoint(workDir, kind, identity) {
  const paths = checkpointPaths(workDir, kind);
  let meta;
  let data;
  try {
    [meta, data] = await Promise.all([readFile(paths.json, "utf8").then(JSON.parse), readFile(paths.bin)]);
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
  const sameRun = ["engine", "model", "rebuild"].every((key) => meta[key] === identity[key]);
  if (!sameRun || data.length !== meta.photoIds.length * embeddingRowBytes) {
    console.log(`Discarding ${kind} checkpoint from a different engine, model, or mode.`);
    return [];
  }
  return meta.photoIds.map((photoId, index) => ({
    photoId,
    row: data.subarray(index * embeddingRowBytes, (index + 1) * embeddingRowBytes),
    textHash: meta.textHashes?.[index] ?? "",
  }));
}

async function writeCheckpoint(workDir, kind, identity, entries) {
  const paths = checkpointPaths(workDir, kind);
  await mkdir(workDir, { recursive: true });
  await writeFile(paths.bin, Buffer.concat(entries.map((entry) => entry.row)));
  await writeFile(paths.json, `${JSON.stringify({
    ...identity,
    kind,
    photoIds: entries.map((entry) => entry.photoId),
    textHashes: entries.map((entry) => entry.textHash ?? ""),
  })}\n`);
}

async function removeCheckpoint(workDir, kind) {
  const paths = checkpointPaths(workDir, kind);
  await rm(paths.bin, { force: true });
  await rm(paths.json, { force: true });
}

// Keeps checkpoint entries that still match the pending work.
function resumeEntries(checkpoint, pending) {
  const pendingById = new Map(pending.map((item) => [item.photoId, item]));
  return checkpoint.filter((entry) => {
    const item = pendingById.get(entry.photoId);
    return item && (item.textHash ?? "") === (entry.textHash ?? "");
  });
}

function formatRate(done, startedAt) {
  const seconds = (Date.now() - startedAt) / 1000;
  return seconds > 0 ? `${(done / seconds).toFixed(1)}/s` : "-";
}

async function embedKind({ kind, pending, client, batchSize, workDir, identity }) {
  const entries = resumeEntries(await readCheckpoint(workDir, kind, identity), pending);
  const doneIds = new Set(entries.map((entry) => entry.photoId));
  const todo = pending.filter((item) => !doneIds.has(item.photoId));
  const total = pending.length;
  if (entries.length > 0) {
    console.log(`${kind}: resuming with ${entries.length}/${total} from checkpoint`);
  }
  if (todo.length === 0) {
    return { entries, failed: [] };
  }

  const thumbsDir = join(workDir, "thumbs");
  if (kind === "image") {
    await mkdir(thumbsDir, { recursive: true });
  }
  const failed = [];
  const shouldReport = createProgressThrottle();
  const startedAt = Date.now();
  let processedThisRun = 0;
  let lastCheckpoint = entries.length;
  const batches = [];
  for (let start = 0; start < todo.length; start += batchSize) {
    batches.push(todo.slice(start, start + batchSize));
  }
  const loadBatchImages = (batch) => mapWithConcurrency(batch, thumbnailConcurrency, (item) => loadThumbnail(item, thumbsDir));

  try {
    let nextImages = kind === "image" ? loadBatchImages(batches[0]) : null;
    for (let batchIndex = 0; batchIndex < batches.length; batchIndex += 1) {
      const batch = batches[batchIndex];
      let items = batch;
      let vectors;
      if (kind === "image") {
        const images = await nextImages;
        // Download the next batch while this one is embedded.
        nextImages = batchIndex + 1 < batches.length ? loadBatchImages(batches[batchIndex + 1]) : null;
        const ready = new Map(images.filter((image) => image.data).map((image) => [image.photoId, image.data]));
        for (const image of images.filter((result) => !result.data)) {
          failed.push(image);
        }
        items = batch.filter((item) => ready.has(item.photoId));
        vectors = items.length ? await client.embedImages(items.map((item) => jpegDataUrl(ready.get(item.photoId)))) : [];
      } else {
        vectors = await client.embedTexts(items.map((item) => item.text));
      }
      items.forEach((item, index) => {
        entries.push({ photoId: item.photoId, row: quantizeEmbedding(vectors[index]), textHash: item.textHash ?? "" });
      });
      processedThisRun += batch.length;

      if (entries.length - lastCheckpoint >= checkpointEvery) {
        lastCheckpoint = entries.length;
        await writeCheckpoint(workDir, kind, identity, entries);
      }
      const done = total - todo.length + processedThisRun;
      if (shouldReport(done, { force: batchIndex === batches.length - 1 })) {
        console.log(`${kind}: ${done}/${total} (${formatRate(processedThisRun, startedAt)})`);
      }
    }
  } catch (error) {
    if (entries.length > 0) {
      await writeCheckpoint(workDir, kind, identity, entries);
      console.error(`${kind}: saved ${entries.length} finished vectors to ${workDir}; rerun to resume.`);
    }
    throw error;
  }

  return { entries, failed };
}

function droppedOnCompact(merged, kind, photoIdSet) {
  return [...merged[kind].keys()].filter((photoId) => !photoIdSet.has(photoId)).length;
}

async function compactStore({ embeddingsDir, kinds, photoIds, createdAt }) {
  const { index, segments } = await readEmbeddingStore(embeddingsDir);
  if (!index || segments.length === 0) {
    console.log("Compact: no segments to compact.");
    return;
  }
  const merged = mergeEmbeddingSegments(segments);
  const existingFiles = new Set(index.segments.map((segment) => segment.file));
  const built = [];
  for (const kind of kinds) {
    if (!segments.some((segment) => segment.kind === kind)) {
      continue;
    }
    const entries = compactEntries({ kind, merged, photoIds });
    if (entries.length > 0) {
      const item = buildEmbeddingSegment({ kind, entries, createdAt, existingFiles });
      existingFiles.add(item.segment.file);
      built.push(item);
    }
    console.log(`Compact ${kind}: ${entries.length} rows kept, ${droppedOnCompact(merged, kind, new Set(photoIds))} dropped`);
  }
  await writeEmbeddingStore({ embeddingsDir, index, built, replaceKinds: kinds });
}

async function main() {
  const options = parseArgs(process.argv);
  if (options.help) {
    printUsage();
    return;
  }

  const kinds = selectedKinds(options.kind);
  const photos = await readPhotos(options);
  const photoIds = photos.map((photo) => String(photo.photo_id));
  const store = await readEmbeddingStore(options.embeddingsDir);
  const merged = mergeEmbeddingSegments(options.rebuild ? [] : store.segments);
  const storedMerged = mergeEmbeddingSegments(store.segments);
  console.log(`Photos: ${photos.length}${options.limit ? ` (limited to first ${options.limit})` : ""}`);
  console.log(`Segments: ${store.segments.length} in ${options.embeddingsDir}`);

  const plans = kinds.map((kind) => ({ kind, ...computeEmbeddingDiff({ kind, merged, photos }) }));
  for (const plan of plans) {
    console.log(`${plan.kind}: ${plan.pending.length} to embed, ${storedMerged[plan.kind].size} stored, ${plan.skipped} skipped (no ${plan.kind === "image" ? "thumbnail URL" : "description or tags"})`);
  }
  if (options.dryRun) {
    if (options.compact) {
      const photoIdSet = new Set(photoIds);
      for (const kind of kinds) {
        console.log(`Compact ${kind}: would drop ${droppedOnCompact(storedMerged, kind, photoIdSet)} rows`);
      }
    }
    console.log("Dry run: no embedding server requests and no files written.");
    return;
  }

  const createdAt = new Date();
  const work = plans.filter((plan) => plan.pending.length > 0);
  if (work.length > 0) {
    const client = createEmbeddingClient({
      engine: options.engine,
      url: options.url,
      model: options.model,
      apiKey: process.env[embeddingApiKeyEnv] ?? "",
    });
    console.log(`Embedding with ${client.engine} ${client.model} at ${client.endpoint}, batch ${options.batch}`);
    const identity = { engine: client.engine, model: client.model, rebuild: options.rebuild };
    let index = store.index;
    for (const plan of work) {
      const { entries, failed } = await embedKind({
        kind: plan.kind,
        pending: plan.pending,
        client,
        batchSize: options.batch,
        workDir: defaultWorkDir,
        identity,
      });
      if (failed.length > 0) {
        console.warn(`${plan.kind}: ${failed.length} thumbnails could not be downloaded; they stay pending for the next run.`);
        for (const item of failed.slice(0, 10)) {
          console.warn(`  ${item.photoId}: ${item.error}`);
        }
      }
      // Persist each kind as soon as it finishes so a later failure keeps it.
      if (entries.length > 0) {
        const existingFiles = new Set((index?.segments ?? []).map((segment) => segment.file));
        const item = buildEmbeddingSegment({ kind: plan.kind, entries, createdAt, existingFiles });
        index = await writeEmbeddingStore({
          embeddingsDir: options.embeddingsDir,
          index,
          built: [item],
          replaceKinds: options.rebuild ? [plan.kind] : [],
        });
        console.log(`Wrote ${join(options.embeddingsDir, item.segment.file)} (${item.segment.count} rows)`);
      }
      await removeCheckpoint(defaultWorkDir, plan.kind);
    }
  } else {
    console.log("No differences; no segment written.");
  }

  if (options.compact) {
    await compactStore({ embeddingsDir: options.embeddingsDir, kinds, photoIds, createdAt });
  }
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  try {
    await main();
  } catch (error) {
    console.error(`Could not build finder embeddings: ${error.message}`);
    process.exitCode = 1;
  }
}
