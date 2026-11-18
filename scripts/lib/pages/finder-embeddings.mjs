import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import {
  buildVectorRows,
  defaultEmbeddingsDir,
  embeddingDims,
  embeddingKinds,
  embeddingQueryModel,
  embeddingQueryPrefix,
  embeddingQueryVocabulary,
  embeddingRowBytes,
  embeddingSourceModel,
  mergeEmbeddingSegments,
  readEmbeddingStore,
  sha256Hex,
} from "../finder/embeddings.mjs";

export const finderVectorsDir = "vectors";

// Verifies the repo's gzipped query vocabulary against the pinned tokenizer.
export function checkQueryVocabulary(compressed) {
  const decoded = gunzipSync(compressed);
  if (decoded.length !== embeddingQueryVocabulary.bytes || sha256Hex(decoded) !== embeddingQueryVocabulary.sha256) {
    throw new Error(`${embeddingQueryVocabulary.file} does not match the pinned query model tokenizer`);
  }
  return { file: embeddingQueryVocabulary.file, encoding: "gzip", bytes: decoded.length, transferBytes: compressed.length, sha256: embeddingQueryVocabulary.sha256 };
}

// The browser receives the repo's immutable segments as they are, so a new
// deployment only costs the segments a visitor does not have yet. Pages
// serves .bin files uncompressed, so each segment ships gzipped (about 25%
// smaller) and is decoded with DecompressionStream; its photo_ids ship as
// JSON. sha256 and bytes describe the decoded rows, transferBytes the .gz.
// `covered` counts photos-index rows that end up with a vector after merging
// (later segments override earlier ones), for the dialog and finder:check.
export const vectorsMetaFormatVersion = "2026-11-vector-segments-v1";

export function segmentArtifactFiles(segment) {
  return { file: `${segment.file}.gz`, ids: segment.ids.replace(/\.json$/, ".ids.json") };
}

// Builds vectors/{meta.json,segments/*} payloads plus the query vocabulary
// when the repo has it. Returns null when there are no embedding segments, so
// the build skips vectors.
export function buildFinderVectorPayloads({ segments, photoIds, vocabulary = null }) {
  if (segments.length === 0) {
    return null;
  }
  const merged = mergeEmbeddingSegments(segments);
  const sets = {};
  const files = {};
  for (const kind of embeddingKinds) {
    const kindSegments = segments.filter((segment) => segment.kind === kind);
    if (kindSegments.length === 0) {
      continue;
    }
    const entries = kindSegments.map((segment) => {
      const names = segmentArtifactFiles(segment);
      const compressed = gzipSync(segment.data, { level: 9 });
      files[names.file] = compressed;
      files[names.ids] = Buffer.from(`${JSON.stringify(segment.photoIds.map(String))}\n`);
      return {
        ...names,
        count: segment.photoIds.length,
        encoding: "gzip",
        bytes: segment.data.length,
        transferBytes: compressed.length,
        idsBytes: files[names.ids].length,
        sha256: sha256Hex(segment.data),
      };
    });
    sets[kind] = {
      covered: buildVectorRows({ kind, merged, photoIds }).covered,
      transferBytes: entries.reduce((total, entry) => total + entry.transferBytes + entry.idsBytes, 0),
      segments: entries,
    };
  }
  const meta = {
    formatVersion: vectorsMetaFormatVersion,
    dims: embeddingDims,
    rowBytes: embeddingRowBytes,
    queryPrefix: embeddingQueryPrefix,
    sourceModel: embeddingSourceModel,
    queryModel: { ...embeddingQueryModel, ...(vocabulary ? { tokenizer: checkQueryVocabulary(vocabulary) } : {}) },
    sets,
  };
  if (vocabulary) {
    files[embeddingQueryVocabulary.file] = vocabulary;
  }
  return { meta, files };
}

export async function buildFinderEmbeddingArtifacts({ embeddingsDir = defaultEmbeddingsDir, finderDataDir, photoIds }) {
  const { segments } = await readEmbeddingStore(embeddingsDir);
  const vocabulary = await readFile(join(embeddingsDir, embeddingQueryVocabulary.file)).catch((error) => {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  });
  const payloads = buildFinderVectorPayloads({ segments, photoIds, vocabulary });
  if (!payloads) {
    return null;
  }
  const outputDir = join(finderDataDir, finderVectorsDir);
  await mkdir(outputDir, { recursive: true });
  for (const [file, data] of Object.entries(payloads.files)) {
    await mkdir(dirname(join(outputDir, file)), { recursive: true });
    await writeFile(join(outputDir, file), data);
  }
  await writeFile(join(outputDir, "meta.json"), `${JSON.stringify(payloads.meta)}\n`);
  return payloads.meta;
}
