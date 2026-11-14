import { mkdir, writeFile } from "node:fs/promises";
import { join } from "node:path";
import {
  buildVectorRows,
  defaultEmbeddingsDir,
  embeddingDims,
  embeddingFormatVersion,
  embeddingKinds,
  embeddingQueryModel,
  embeddingQueryPrefix,
  embeddingRowBytes,
  mergeEmbeddingSegments,
  readEmbeddingStore,
  sha256Hex,
} from "../finder/embeddings.mjs";

export const finderVectorsDir = "vectors";

// Builds vectors/{meta.json,<kind>-int8.bin} payloads in photos-index row order.
// Returns null when there are no embedding segments, so the build skips vectors.
export function buildFinderVectorPayloads({ segments, photoIds, photosSha256 }) {
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
    const file = `${kind}-int8.bin`;
    const { data, covered } = buildVectorRows({ kind, merged, photoIds });
    sets[kind] = { file, bytes: data.length, sha256: sha256Hex(data), covered };
    files[file] = data;
  }
  const meta = {
    formatVersion: embeddingFormatVersion,
    dims: embeddingDims,
    rowBytes: embeddingRowBytes,
    rowCount: photoIds.length,
    queryPrefix: embeddingQueryPrefix,
    queryModel: { ...embeddingQueryModel },
    photosSha256,
    sets,
  };
  return { meta, files };
}

export async function buildFinderEmbeddingArtifacts({ embeddingsDir = defaultEmbeddingsDir, finderDataDir, photoIds, photosSha256 }) {
  const { segments } = await readEmbeddingStore(embeddingsDir);
  const payloads = buildFinderVectorPayloads({ segments, photoIds, photosSha256 });
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
