import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { gunzipSync, gzipSync } from "node:zlib";
import {
  buildVectorRows,
  defaultEmbeddingsDir,
  embeddingDims,
  embeddingFormatVersion,
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

// Pages serves .bin files uncompressed, so vector sets ship gzipped (about 25%
// smaller) and the browser decodes them with DecompressionStream. sha256 and
// bytes always describe the decoded data; transferBytes is the .gz size.
function gzipEntry(file, data) {
  const compressed = gzipSync(data, { level: 9 });
  return { entry: { file, encoding: "gzip", bytes: data.length, transferBytes: compressed.length, sha256: sha256Hex(data) }, compressed };
}

// Verifies the repo's gzipped query vocabulary against the pinned tokenizer.
export function checkQueryVocabulary(compressed) {
  const decoded = gunzipSync(compressed);
  if (decoded.length !== embeddingQueryVocabulary.bytes || sha256Hex(decoded) !== embeddingQueryVocabulary.sha256) {
    throw new Error(`${embeddingQueryVocabulary.file} does not match the pinned query model tokenizer`);
  }
  return { file: embeddingQueryVocabulary.file, encoding: "gzip", bytes: decoded.length, transferBytes: compressed.length, sha256: embeddingQueryVocabulary.sha256 };
}

// Builds vectors/{meta.json,<kind>-int8.bin.gz} payloads in photos-index row
// order, plus the query vocabulary when the repo has it. Returns null when
// there are no embedding segments, so the build skips vectors.
export function buildFinderVectorPayloads({ segments, photoIds, photosSha256, vocabulary = null }) {
  if (segments.length === 0) {
    return null;
  }
  const merged = mergeEmbeddingSegments(segments);
  const sets = {};
  const files = {};
  for (const kind of embeddingKinds) {
    if (!segments.some((segment) => segment.kind === kind)) {
      continue;
    }
    const { data, covered } = buildVectorRows({ kind, merged, photoIds });
    const { entry, compressed } = gzipEntry(`${kind}-int8.bin.gz`, data);
    sets[kind] = { ...entry, covered };
    files[entry.file] = compressed;
  }
  const meta = {
    formatVersion: embeddingFormatVersion,
    dims: embeddingDims,
    rowBytes: embeddingRowBytes,
    rowCount: photoIds.length,
    queryPrefix: embeddingQueryPrefix,
    sourceModel: embeddingSourceModel,
    queryModel: { ...embeddingQueryModel, ...(vocabulary ? { tokenizer: checkQueryVocabulary(vocabulary) } : {}) },
    photosSha256,
    sets,
  };
  if (vocabulary) {
    files[embeddingQueryVocabulary.file] = vocabulary;
  }
  return { meta, files };
}

export async function buildFinderEmbeddingArtifacts({ embeddingsDir = defaultEmbeddingsDir, finderDataDir, photoIds, photosSha256 }) {
  const { segments } = await readEmbeddingStore(embeddingsDir);
  const vocabulary = await readFile(join(embeddingsDir, embeddingQueryVocabulary.file)).catch((error) => {
    if (error.code === "ENOENT") {
      return null;
    }
    throw error;
  });
  const payloads = buildFinderVectorPayloads({ segments, photoIds, photosSha256, vocabulary });
  if (!payloads) {
    return null;
  }
  const outputDir = join(finderDataDir, finderVectorsDir);
  await mkdir(outputDir, { recursive: true });
  for (const [file, data] of Object.entries(payloads.files)) {
    await writeFile(join(outputDir, file), data);
  }
  await writeFile(join(outputDir, "meta.json"), `${JSON.stringify(payloads.meta)}\n`);
  return payloads.meta;
}
