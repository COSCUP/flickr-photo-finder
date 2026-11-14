// Pure ranking helpers for opt-in semantic search and image similarity.
// This module must not touch the DOM, network, or storage; callers pass parsed
// vector sets, query vectors, and eligibility masks explicitly.

export const vectorFormatVersion = "2026-11-embeddings-v1";
export const vectorSetKinds = ["text", "image"];

// Sort keys pack a quantized score and the row index into one float64 so that
// native TypedArray sorting can rank rows without per-comparison callbacks.
const indexRadix = 2 ** 20;
const cosineScale = 2 ** 31;
const rrfScale = 2 ** 32;
export const maxVectorRows = indexRadix;

function positiveInteger(value) {
  const number = Number(value);
  return Number.isInteger(number) && number > 0 ? number : 0;
}

export function parseVectorMeta(payload) {
  if (!payload || typeof payload !== "object") {
    throw new Error("向量 meta 格式不正確");
  }
  const dims = positiveInteger(payload.dims);
  const rowBytes = positiveInteger(payload.rowBytes);
  const rowCount = positiveInteger(payload.rowCount);
  if (payload.formatVersion !== vectorFormatVersion) {
    throw new Error(`不支援的向量格式：${payload.formatVersion ?? "未提供"}`);
  }
  if (!dims || rowBytes !== dims + 4) {
    throw new Error("向量維度與每列大小不一致");
  }
  if (!rowCount || rowCount > maxVectorRows) {
    throw new Error("向量列數不正確");
  }
  const sets = {};
  for (const kind of vectorSetKinds) {
    const set = payload.sets?.[kind];
    if (!set) {
      continue;
    }
    const bytes = positiveInteger(set.bytes);
    if (!set.file || bytes !== rowCount * rowBytes || !/^[0-9a-f]{64}$/.test(String(set.sha256 ?? ""))) {
      throw new Error(`${kind} 向量檔資訊不正確`);
    }
    sets[kind] = {
      kind,
      file: String(set.file),
      bytes,
      sha256: String(set.sha256),
      covered: Number(set.covered) || 0,
    };
  }
  return {
    formatVersion: payload.formatVersion,
    dims,
    rowBytes,
    rowCount,
    queryPrefix: String(payload.queryPrefix ?? ""),
    sourceModel: String(payload.sourceModel ?? ""),
    queryModel: {
      id: String(payload.queryModel?.id ?? ""),
      revision: String(payload.queryModel?.revision ?? ""),
      dtype: String(payload.queryModel?.dtype ?? ""),
    },
    photosSha256: String(payload.photosSha256 ?? ""),
    sets,
  };
}

// Parses one quantized vector file: each row is `dims` int8 values followed by
// a little-endian float32 scale. A zero scale marks a missing vector. Cosine
// scores do not need the scale because each row is divided by its own norm.
export function parseVectorSet(buffer, { dims, rowBytes, rowCount }) {
  const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
  if (bytes.byteLength !== rowCount * rowBytes) {
    throw new Error(`向量檔大小不符：預期 ${rowCount * rowBytes} bytes，實際 ${bytes.byteLength} bytes`);
  }
  const values = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const inverseNorms = new Float32Array(rowCount);
  let covered = 0;
  for (let row = 0; row < rowCount; row += 1) {
    const base = row * rowBytes;
    const scale = view.getFloat32(base + dims, true);
    if (!(scale > 0)) {
      continue;
    }
    let sum = 0;
    for (let index = 0; index < dims; index += 1) {
      const value = values[base + index];
      sum += value * value;
    }
    if (sum > 0) {
      inverseNorms[row] = 1 / Math.sqrt(sum);
      covered += 1;
    }
  }
  return { dims, rowBytes, rowCount, values, inverseNorms, covered };
}

export function hasVector(set, row) {
  return Boolean(set) && row >= 0 && row < set.rowCount && set.inverseNorms[row] > 0;
}

export function normalizeVector(vector, dims = vector.length) {
  const length = Math.min(dims, vector.length);
  const output = new Float32Array(length);
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    sum += vector[index] * vector[index];
  }
  const inverse = sum > 0 ? 1 / Math.sqrt(sum) : 0;
  for (let index = 0; index < length; index += 1) {
    output[index] = vector[index] * inverse;
  }
  return output;
}

export function rowVector(set, row) {
  if (!hasVector(set, row)) {
    return null;
  }
  const output = new Float32Array(set.dims);
  const base = row * set.rowBytes;
  const inverse = set.inverseNorms[row];
  for (let index = 0; index < set.dims; index += 1) {
    output[index] = set.values[base + index] * inverse;
  }
  return output;
}

// Reusable scratch buffers, keyed by row count, so repeated searches do not
// allocate tens of megabytes per keystroke.
const scratch = { rowCount: 0, scores: null, keys: null, fused: null, ranked: [] };

function scratchFor(rowCount) {
  if (scratch.rowCount !== rowCount) {
    scratch.rowCount = rowCount;
    scratch.scores = new Float32Array(rowCount);
    scratch.keys = new Float64Array(rowCount);
    scratch.fused = new Float64Array(rowCount);
    scratch.ranked = [new Uint32Array(rowCount), new Uint32Array(rowCount)];
  }
  return scratch;
}

// Cosine score for every row with a vector; rows without one get -Infinity.
export function scoreRows(set, query, output = new Float32Array(set.rowCount)) {
  const { dims, rowBytes, rowCount, values, inverseNorms } = set;
  if (query.length < dims) {
    throw new Error("查詢向量維度不足");
  }
  const limit = dims - (dims % 4);
  for (let row = 0; row < rowCount; row += 1) {
    const inverse = inverseNorms[row];
    if (inverse === 0) {
      output[row] = Number.NEGATIVE_INFINITY;
      continue;
    }
    const base = row * rowBytes;
    let sum0 = 0;
    let sum1 = 0;
    let sum2 = 0;
    let sum3 = 0;
    let index = 0;
    for (; index < limit; index += 4) {
      sum0 += values[base + index] * query[index];
      sum1 += values[base + index + 1] * query[index + 1];
      sum2 += values[base + index + 2] * query[index + 2];
      sum3 += values[base + index + 3] * query[index + 3];
    }
    for (; index < dims; index += 1) {
      sum0 += values[base + index] * query[index];
    }
    output[row] = (sum0 + sum1 + sum2 + sum3) * inverse;
  }
  return output;
}

function isEligible(eligible, row) {
  return !eligible || eligible[row] === 1;
}

// Ranks eligible rows with a finite score, best first. Returns the number of
// ranked rows written into `output`.
export function rankScores(scores, { eligible = null, excludeRow = -1, keys = new Float64Array(scores.length), output = new Uint32Array(scores.length) } = {}) {
  let count = 0;
  for (let row = 0; row < scores.length; row += 1) {
    const score = scores[row];
    if (score === Number.NEGATIVE_INFINITY || row === excludeRow || !isEligible(eligible, row)) {
      continue;
    }
    const clamped = Math.min(1, Math.max(-1, score));
    keys[count] = Math.round((1 - clamped) * cosineScale) * indexRadix + row;
    count += 1;
  }
  const sorted = keys.subarray(0, count).sort();
  for (let index = 0; index < count; index += 1) {
    output[index] = sorted[index] % indexRadix;
  }
  return count;
}

export function topRows(scores, limit, options = {}) {
  const output = options.output ?? new Uint32Array(scores.length);
  const count = rankScores(scores, { ...options, output });
  return Array.from(output.subarray(0, Math.min(limit, count)));
}

// Reciprocal rank fusion over full rankings: each list adds 1 / (k + rank),
// with rank starting at 1. Rows absent from a list get no contribution.
export function fuseRankings(rankings, { rowCount, k = 60, limit = Infinity, fused = new Float64Array(rowCount), keys = new Float64Array(rowCount) } = {}) {
  fused.fill(0, 0, rowCount);
  for (const ranking of rankings) {
    for (let position = 0; position < ranking.length; position += 1) {
      fused[ranking[position]] += 1 / (k + position + 1);
    }
  }
  let count = 0;
  for (let row = 0; row < rowCount; row += 1) {
    if (fused[row] > 0) {
      keys[count] = Math.round((1 - fused[row]) * rrfScale) * indexRadix + row;
      count += 1;
    }
  }
  const sorted = keys.subarray(0, count).sort();
  const length = Math.min(limit, count);
  const output = new Array(length);
  for (let index = 0; index < length; index += 1) {
    output[index] = sorted[index] % indexRadix;
  }
  return output;
}

function rowCountOf(sets) {
  const counts = sets.filter(Boolean).map((set) => set.rowCount);
  if (counts.length === 0) {
    return 0;
  }
  if (counts.some((count) => count !== counts[0])) {
    throw new Error("向量集的列數不一致");
  }
  return counts[0];
}

// Text query ranking: score the query against every available vector set,
// rank eligible rows per set, then fuse with RRF. Rows missing every vector
// never appear. Returns row indices, best first, at most `topK`.
export function semanticRanking({ sets = {}, query, eligible = null, topK = 200, rrfK = 60 }) {
  const activeSets = vectorSetKinds.map((kind) => sets[kind]).filter(Boolean);
  const rowCount = rowCountOf(activeSets);
  if (!query || rowCount === 0) {
    return [];
  }
  const buffers = scratchFor(rowCount);
  const rankings = activeSets.map((set, index) => {
    scoreRows(set, query, buffers.scores);
    const count = rankScores(buffers.scores, { eligible, keys: buffers.keys, output: buffers.ranked[index] });
    return buffers.ranked[index].subarray(0, count);
  });
  if (rankings.length === 1) {
    return Array.from(rankings[0].subarray(0, Math.min(topK, rankings[0].length)));
  }
  return fuseRankings(rankings, { rowCount, k: rrfK, limit: topK, fused: buffers.fused, keys: buffers.keys });
}

// Image similarity: rank by image-vector cosine against the source row only,
// excluding the source photo itself.
export function similarRanking({ image, sourceRow, eligible = null, topK = 200 }) {
  const query = rowVector(image, sourceRow);
  if (!query) {
    return [];
  }
  const buffers = scratchFor(image.rowCount);
  scoreRows(image, query, buffers.scores);
  const count = rankScores(buffers.scores, { eligible, excludeRow: sourceRow, keys: buffers.keys, output: buffers.ranked[0] });
  return Array.from(buffers.ranked[0].subarray(0, Math.min(topK, count)));
}

export function eligibilityMask(items, predicate, output = new Uint8Array(items.length)) {
  for (let index = 0; index < items.length; index += 1) {
    output[index] = predicate(items[index], index) ? 1 : 0;
  }
  return output;
}

// Decides which result list is shown. `similar` wins when image vectors are
// ready; otherwise a text query with ready text search gets the tabbed view.
// Shared links that ask for semantic views fall back to keyword results when
// this device has not enabled the feature.
export function resolveResultView({ textReady = false, imageReady = false, query = "", similarPhotoId = "", requestedView = "" } = {}) {
  if (similarPhotoId && imageReady) {
    return { mode: "similar", showTabs: false };
  }
  if (textReady && String(query ?? "").trim()) {
    return { mode: requestedView === "keyword" ? "keyword" : "semantic", showTabs: true };
  }
  return { mode: "keyword", showTabs: false };
}

// Entering "similar to" remembers the query and tab it replaced, so removing
// the condition returns to them. Inputs and outputs are plain objects:
// { search, view, similar, similarReturn }.
export function enterSimilarState(current, photoId) {
  const similarReturn = current.similar && current.similarReturn
    ? current.similarReturn
    : { search: String(current.search ?? ""), view: String(current.view ?? "") };
  return { search: "", view: current.view ?? "", similar: String(photoId), similarReturn };
}

export function exitSimilarState(current) {
  if (!current.similarReturn) {
    return { search: String(current.search ?? ""), view: current.view ?? "", similar: "", similarReturn: null };
  }
  return { search: current.similarReturn.search, view: current.similarReturn.view, similar: "", similarReturn: null };
}

// Text queries need WebGPU: every quantized EmbeddingGemma 2 export uses
// GatherBlockQuantized, which the ORT-web WASM backend does not implement.
// Image similarity needs only the image vectors. Returns the mode a stored
// preference can actually run in on this device. A degraded mode is
// in-memory only: `storedPreference` keeps "full" so the same profile restores
// full search once WebGPU is available again.
export function resolveSemanticMode({ preference = "", webgpu = false, imageCached = false } = {}) {
  if (preference === "full") {
    if (webgpu) {
      return { mode: "full", reason: "", storedPreference: "full" };
    }
    return { mode: imageCached ? "image" : "", reason: "no-webgpu", storedPreference: "full" };
  }
  if (preference === "image") {
    return { mode: "image", reason: "", storedPreference: "image" };
  }
  return { mode: "", reason: "", storedPreference: "" };
}

export function photosForRows(photos, rows) {
  return rows.map((row) => photos[row]).filter(Boolean);
}

export function splitResultTabs({ photos, keywordResults, semanticRows }) {
  return {
    keyword: keywordResults,
    semantic: photosForRows(photos, semanticRows),
  };
}
