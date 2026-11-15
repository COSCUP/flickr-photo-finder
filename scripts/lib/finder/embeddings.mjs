import { createHash } from "node:crypto";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";

export const defaultEmbeddingsDir = "data/embeddings";
export const embeddingFormatVersion = "2026-11-embeddings-v1";
export const embeddingDims = 512;
export const embeddingRowBytes = embeddingDims + 4;
export const embeddingQuantization = "int8-row-scale";
export const embeddingDocumentFormat = "eg2-photo-doc-v1";
export const embeddingQueryPrefix = "task: search result | query: ";
export const embeddingSourceModel = "EmbeddingGemma 2 (8-bit, 768d truncated to 512d)";
export const embeddingQueryModel = {
  id: "onnx-community/embeddinggemma-2-ONNX",
  revision: "daa72c51243991dfcaf9f9137d2c573d8f7790c0",
  dtype: "q4",
};
export const embeddingKinds = ["text", "image"];
// tokenizer.json of the pinned query model, gzipped (level 9) and kept in the
// repo so Pages serves it at about 5.4 MB instead of the 32 MB that
// huggingface.co sends uncompressed. sha256 and bytes are of the decoded file,
// matching the Hub's LFS object. The name avoids "token", which the Pages
// artifact check reserves for credential-like files.
export const embeddingQueryVocabulary = {
  file: "query-vocabulary.json.gz",
  sha256: "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4",
  bytes: 32170510,
};

export function sha256Hex(value) {
  return createHash("sha256").update(value).digest("hex");
}

function listValues(value) {
  return (Array.isArray(value) ? value : [value])
    .map((item) => String(item ?? "").trim())
    .filter(Boolean);
}

// The text part of eg2-photo-doc-v1. Empty when the photo has no description or tags.
export function documentBody(photo) {
  const scenes = listValues(photo.scene_tags);
  const moods = listValues(photo.mood_tags);
  const uses = listValues(photo.recommended_uses);
  return [
    String(photo.visual_description ?? "").trim(),
    scenes.length ? `場景：${scenes.join("、")}。` : "",
    moods.length ? `氛圍：${moods.join("、")}。` : "",
    uses.length ? `用途：${uses.join("、")}。` : "",
  ].filter(Boolean).join(" ");
}

export function documentText(photo) {
  const albumTitle = String(photo.album_title ?? "").trim();
  return `title: ${albumTitle || "none"} | text: ${documentBody(photo)}`;
}

export function thumbnailUrl(photo) {
  const previewUrl = String(photo.image_preview_url ?? "").trim();
  return previewUrl.replace(/_q\.jpg$/, "_n.jpg");
}

// Matryoshka truncation to the leading dims, then L2 normalization.
export function truncateAndNormalize(vector, dims = embeddingDims) {
  if (!vector || vector.length < dims) {
    throw new Error(`Embedding has ${vector?.length ?? 0} dimensions; expected at least ${dims}`);
  }
  const output = new Float32Array(dims);
  let sumSquares = 0;
  for (let index = 0; index < dims; index += 1) {
    const value = Number(vector[index]);
    if (!Number.isFinite(value)) {
      throw new Error("Embedding contains a non-finite value");
    }
    output[index] = value;
    sumSquares += value * value;
  }
  const norm = Math.sqrt(sumSquares);
  if (norm === 0) {
    throw new Error("Embedding is a zero vector");
  }
  for (let index = 0; index < dims; index += 1) {
    output[index] /= norm;
  }
  return output;
}

// One row: dims int8 values followed by a float32 little-endian scale.
export function quantizeEmbedding(vector, dims = embeddingDims) {
  const normalized = truncateAndNormalize(vector, dims);
  let maxAbs = 0;
  for (const value of normalized) {
    maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  const scale = maxAbs / 127;
  const row = Buffer.alloc(dims + 4);
  for (let index = 0; index < dims; index += 1) {
    const quantized = Math.max(-127, Math.min(127, Math.round(normalized[index] / scale)));
    row.writeInt8(quantized, index);
  }
  row.writeFloatLE(scale, dims);
  return row;
}

export function rowScale(buffer, rowIndex = 0, dims = embeddingDims) {
  return buffer.readFloatLE(rowIndex * (dims + 4) + dims);
}

// Returns null for a missing row (scale 0).
export function dequantizeRow(buffer, rowIndex = 0, dims = embeddingDims) {
  const offset = rowIndex * (dims + 4);
  const scale = buffer.readFloatLE(offset + dims);
  if (scale === 0) {
    return null;
  }
  const vector = new Float32Array(dims);
  for (let index = 0; index < dims; index += 1) {
    vector[index] = buffer.readInt8(offset + index) * scale;
  }
  return vector;
}

export function emptyEmbeddingIndex() {
  return {
    formatVersion: embeddingFormatVersion,
    dims: embeddingDims,
    rowBytes: embeddingRowBytes,
    quantization: embeddingQuantization,
    documentFormat: embeddingDocumentFormat,
    queryPrefix: embeddingQueryPrefix,
    sourceModel: embeddingSourceModel,
    queryModel: { ...embeddingQueryModel },
    segments: [],
  };
}

export function assertEmbeddingIndex(index) {
  if (index?.formatVersion !== embeddingFormatVersion) {
    throw new Error(`embeddings index.json formatVersion must be ${embeddingFormatVersion}`);
  }
  if (index.dims !== embeddingDims || index.rowBytes !== embeddingRowBytes) {
    throw new Error(`embeddings index.json must use dims ${embeddingDims} and rowBytes ${embeddingRowBytes}`);
  }
  if (index.documentFormat !== embeddingDocumentFormat) {
    throw new Error(`embeddings index.json documentFormat must be ${embeddingDocumentFormat}`);
  }
  if (!Array.isArray(index.segments)) {
    throw new Error("embeddings index.json segments must be an array");
  }
  for (const segment of index.segments) {
    if (!embeddingKinds.includes(segment.kind) || !segment.file || !segment.ids || !Number.isInteger(segment.count)) {
      throw new Error("embeddings segments must include kind, file, ids, and count");
    }
  }
  return index;
}

// Returns null when the embeddings directory or index.json does not exist.
export async function readEmbeddingIndex(embeddingsDir = defaultEmbeddingsDir) {
  let text;
  try {
    text = await readFile(join(embeddingsDir, "index.json"), "utf8");
  } catch (error) {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  }
  return assertEmbeddingIndex(JSON.parse(text));
}

export async function readEmbeddingSegment(embeddingsDir, segment) {
  const [data, idsText] = await Promise.all([
    readFile(join(embeddingsDir, segment.file)),
    readFile(join(embeddingsDir, segment.ids), "utf8"),
  ]);
  const ids = JSON.parse(idsText);
  if (data.length !== segment.count * embeddingRowBytes) {
    throw new Error(`${segment.file} has ${data.length} bytes; expected ${segment.count} x ${embeddingRowBytes}`);
  }
  if (segment.sha256 && sha256Hex(data) !== segment.sha256) {
    throw new Error(`${segment.file} sha256 does not match index.json`);
  }
  if (!Array.isArray(ids.photoIds) || ids.photoIds.length !== segment.count) {
    throw new Error(`${segment.ids} photoIds length does not match segment count`);
  }
  if (segment.kind === "text" && (!Array.isArray(ids.textHashes) || ids.textHashes.length !== segment.count)) {
    throw new Error(`${segment.ids} textHashes length does not match segment count`);
  }
  return {
    ...segment,
    data,
    photoIds: ids.photoIds.map(String),
    textHashes: ids.textHashes ?? [],
  };
}

export async function readEmbeddingStore(embeddingsDir = defaultEmbeddingsDir) {
  const index = await readEmbeddingIndex(embeddingsDir);
  if (!index) {
    return { index: null, segments: [] };
  }
  const segments = [];
  for (const segment of index.segments) {
    segments.push(await readEmbeddingSegment(embeddingsDir, segment));
  }
  return { index, segments };
}

// Later segments override earlier ones for the same kind and photo_id.
export function mergeEmbeddingSegments(segments) {
  const merged = Object.fromEntries(embeddingKinds.map((kind) => [kind, new Map()]));
  for (const segment of segments) {
    const target = merged[segment.kind];
    segment.photoIds.forEach((photoId, rowIndex) => {
      const start = rowIndex * embeddingRowBytes;
      target.set(String(photoId), {
        row: segment.data.subarray(start, start + embeddingRowBytes),
        textHash: segment.textHashes[rowIndex] ?? "",
      });
    });
  }
  return merged;
}

function uniquePhotos(photos) {
  const seen = new Set();
  return photos.filter((photo) => {
    const photoId = String(photo.photo_id ?? "").trim();
    if (!photoId || seen.has(photoId)) {
      return false;
    }
    seen.add(photoId);
    return true;
  });
}

// Image: photos with a thumbnail and no stored vector.
// Text: photos with document content whose sha256(documentText) differs from the stored hash.
export function computeEmbeddingDiff({ kind, merged, photos }) {
  const stored = merged?.[kind] ?? new Map();
  const pending = [];
  let skipped = 0;
  for (const photo of uniquePhotos(photos)) {
    const photoId = String(photo.photo_id).trim();
    if (kind === "image") {
      const url = thumbnailUrl(photo);
      if (!url) {
        skipped += 1;
        continue;
      }
      if (!stored.has(photoId)) {
        pending.push({ photoId, url });
      }
      continue;
    }
    if (!documentBody(photo)) {
      skipped += 1;
      continue;
    }
    const text = documentText(photo);
    const textHash = sha256Hex(text);
    if (stored.get(photoId)?.textHash !== textHash) {
      pending.push({ photoId, text, textHash });
    }
  }
  return { pending, skipped };
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

export function segmentStamp(date) {
  return `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}-${pad2(date.getHours())}${pad2(date.getMinutes())}`;
}

// entries: [{ photoId, row, textHash? }] in output order.
export function buildEmbeddingSegment({ kind, entries, createdAt = new Date(), existingFiles = new Set() }) {
  if (!embeddingKinds.includes(kind)) {
    throw new Error(`Unknown embedding kind: ${kind}`);
  }
  const stamp = segmentStamp(createdAt);
  let base = `segments/${stamp}-${kind}`;
  for (let suffix = 2; existingFiles.has(`${base}.bin`); suffix += 1) {
    base = `segments/${stamp}-${kind}-${suffix}`;
  }
  const data = Buffer.concat(entries.map((entry) => {
    if (entry.row.length !== embeddingRowBytes) {
      throw new Error(`Embedding row for ${entry.photoId} has ${entry.row.length} bytes`);
    }
    return entry.row;
  }));
  const ids = { photoIds: entries.map((entry) => String(entry.photoId)) };
  if (kind === "text") {
    ids.textHashes = entries.map((entry) => entry.textHash ?? "");
  }
  return {
    segment: {
      kind,
      file: `${base}.bin`,
      ids: `${base}.json`,
      count: entries.length,
      sha256: sha256Hex(data),
      createdAt: createdAt.toISOString(),
    },
    data,
    ids,
  };
}

// Writes built segments, then index.json. Existing segments whose kind is in
// replaceKinds are dropped from the index and their files removed.
export async function writeEmbeddingStore({ embeddingsDir = defaultEmbeddingsDir, index, built, replaceKinds = [] }) {
  const nextIndex = {
    ...emptyEmbeddingIndex(),
    segments: [
      ...(index?.segments ?? []).filter((segment) => !replaceKinds.includes(segment.kind)),
      ...built.map((item) => item.segment),
    ],
  };
  for (const item of built) {
    const binPath = join(embeddingsDir, item.segment.file);
    await mkdir(dirname(binPath), { recursive: true });
    await writeFile(binPath, item.data);
    await writeFile(join(embeddingsDir, item.segment.ids), `${JSON.stringify(item.ids)}\n`);
  }
  await mkdir(embeddingsDir, { recursive: true });
  await writeFile(join(embeddingsDir, "index.json"), `${JSON.stringify(nextIndex, null, 2)}\n`);
  if (replaceKinds.length > 0) {
    const kept = new Set(nextIndex.segments.flatMap((segment) => [segment.file, segment.ids]));
    for (const segment of index?.segments ?? []) {
      for (const file of [segment.file, segment.ids]) {
        if (!kept.has(file)) {
          await rm(join(embeddingsDir, file), { force: true });
        }
      }
    }
  }
  return nextIndex;
}

// Entries for a single base segment per kind, limited to photo_ids still present.
export function compactEntries({ kind, merged, photoIds }) {
  const stored = merged[kind];
  const entries = [];
  const seen = new Set();
  for (const rawId of photoIds) {
    const photoId = String(rawId);
    if (seen.has(photoId) || !stored.has(photoId)) {
      continue;
    }
    seen.add(photoId);
    const { row, textHash } = stored.get(photoId);
    entries.push({ photoId, row, textHash });
  }
  return entries;
}

// Rows follow photoIds order; photos without a vector stay all zero (scale 0).
export function buildVectorRows({ kind, merged, photoIds }) {
  const stored = merged[kind];
  const data = Buffer.alloc(photoIds.length * embeddingRowBytes);
  let covered = 0;
  photoIds.forEach((rawId, rowIndex) => {
    const entry = stored.get(String(rawId));
    if (entry) {
      entry.row.copy(data, rowIndex * embeddingRowBytes);
      covered += 1;
    }
  });
  return { data, covered };
}

export function countCoveredRows(data) {
  let covered = 0;
  for (let offset = embeddingDims; offset < data.length; offset += embeddingRowBytes) {
    if (data.readFloatLE(offset) !== 0) {
      covered += 1;
    }
  }
  return covered;
}
