// Pure helpers for opt-in semantic search: ranking and image similarity, plus
// the query-model backend choice and ONNX graph rewrite used by the worker.
// This module must not touch the DOM, network, or storage; callers pass parsed
// vector sets, query vectors, model bytes and eligibility masks explicitly.

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
      ...encodedTransfer(set, bytes),
      sha256: String(set.sha256),
      covered: Number(set.covered) || 0,
    };
  }
  const tokenizer = parseEncodedFile(payload.queryModel?.tokenizer);
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
      ...(tokenizer ? { tokenizer } : {}),
    },
    photosSha256: String(payload.photosSha256 ?? ""),
    sets,
  };
}

// Files may ship gzip-encoded; `bytes` and `sha256` always describe the decoded
// data and `transferBytes` what is downloaded.
function encodedTransfer(entry, bytes) {
  const encoding = entry.encoding === "gzip" ? "gzip" : "";
  return { encoding, transferBytes: encoding ? positiveInteger(entry.transferBytes) || bytes : bytes };
}

function parseEncodedFile(entry) {
  if (!entry || typeof entry !== "object") {
    return null;
  }
  const bytes = positiveInteger(entry.bytes);
  if (!entry.file || !bytes || !/^[0-9a-f]{64}$/.test(String(entry.sha256 ?? ""))) {
    return null;
  }
  return { file: String(entry.file), bytes, ...encodedTransfer(entry, bytes), sha256: String(entry.sha256) };
}

// Reads a response body into exactly `expectedBytes` bytes. A gzip body (Pages
// serves .gz files as-is) is decoded with DecompressionStream; a body that a
// proxy already decoded is detected by its missing gzip magic and used as is.
// onProgress receives the number of body bytes received so far.
export async function readEncodedBody(response, { expectedBytes, onProgress = () => {} }) {
  const output = new Uint8Array(expectedBytes);
  let offset = 0;
  const append = (chunk) => {
    if (offset + chunk.byteLength > expectedBytes) {
      throw new Error("下載的檔案大小超出預期");
    }
    output.set(chunk, offset);
    offset += chunk.byteLength;
  };
  const source = response.body ?? new Response(await response.arrayBuffer()).body;
  const reader = source.getReader();
  let received = 0;
  const first = await reader.read();
  const head = first.done ? new Uint8Array(0) : first.value;
  received += head.byteLength;
  onProgress(received);
  const gzip = head.byteLength >= 2 && head[0] === 0x1f && head[1] === 0x8b;
  if (!gzip) {
    append(head);
    for (let chunk = first.done ? { done: true } : await reader.read(); !chunk.done; chunk = await reader.read()) {
      received += chunk.value.byteLength;
      onProgress(received);
      append(chunk.value);
    }
  } else {
    const counted = new ReadableStream({
      start(controller) {
        controller.enqueue(head);
      },
      async pull(controller) {
        const chunk = await reader.read();
        if (chunk.done) {
          controller.close();
          return;
        }
        received += chunk.value.byteLength;
        onProgress(received);
        controller.enqueue(chunk.value);
      },
      cancel(reason) {
        return reader.cancel(reason);
      },
    });
    const decoded = counted.pipeThrough(new DecompressionStream("gzip")).getReader();
    for (let chunk = await decoded.read(); !chunk.done; chunk = await decoded.read()) {
      try {
        append(chunk.value);
      } catch (error) {
        await decoded.cancel();
        throw error;
      }
    }
  }
  if (offset !== expectedBytes) {
    throw new Error("檔案下載不完整");
  }
  return output;
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

export function photosForRows(photos, rows) {
  return rows.map((row) => photos[row]).filter(Boolean);
}

export function splitResultTabs({ photos, keywordResults, semanticRows }) {
  return {
    keyword: keywordResults,
    semantic: photosForRows(photos, semanticRows),
  };
}

// Where the query model runs. WebGPU is fastest where it works well (Apple
// GPUs: about 25 ms per query). Without a WebGPU adapter, and on GPUs whose
// WebGPU matrix multiplication is slow, the WASM CPU backend is used: on a
// Pixel 8 Pro (Mali, vendor "arm") a query took about 10 s on WebGPU and
// under 1 s on WASM. `override` ("webgpu" or "wasm") is a debug switch.
export const wasmPreferredGpuVendors = ["arm"];

export function chooseQueryBackend({ hasWebGpu = false, vendor = "", override = "" } = {}) {
  if (override === "wasm" || (override === "webgpu" && hasWebGpu)) {
    return override;
  }
  if (!hasWebGpu || wasmPreferredGpuVendors.includes(String(vendor).toLowerCase())) {
    return "wasm";
  }
  return "webgpu";
}

// Minimal protobuf helpers for ONNX ModelProto bytes.
function readVarint(view, position) {
  let value = 0;
  let shift = 0;
  for (;;) {
    if (position >= view.length) {
      throw new Error("ONNX 模型格式不正確");
    }
    const byte = view[position++];
    value += (byte & 0x7f) * 2 ** shift;
    if (byte < 0x80) {
      return [value, position];
    }
    shift += 7;
  }
}

function writeVarint(value) {
  const out = [];
  if (value < 0) {
    let big = BigInt.asUintN(64, BigInt(value));
    while (big >= 0x80n) {
      out.push(Number(big & 0x7fn) | 0x80);
      big >>= 7n;
    }
    out.push(Number(big));
    return out;
  }
  let rest = value;
  while (rest >= 0x80) {
    out.push((rest % 128) | 0x80);
    rest = Math.floor(rest / 128);
  }
  out.push(rest);
  return out;
}

function protoFields(view) {
  const list = [];
  let position = 0;
  while (position < view.length) {
    const start = position;
    let key;
    [key, position] = readVarint(view, position);
    const field = Math.floor(key / 8);
    const wire = key % 8;
    let payloadStart = position;
    let value = 0;
    if (wire === 0) {
      [value, position] = readVarint(view, position);
    } else if (wire === 1) {
      position += 8;
    } else if (wire === 5) {
      position += 4;
    } else if (wire === 2) {
      let length;
      [length, position] = readVarint(view, position);
      payloadStart = position;
      position += length;
    } else {
      throw new Error("ONNX 模型格式不正確");
    }
    if (position > view.length) {
      throw new Error("ONNX 模型格式不正確");
    }
    list.push({ field, wire, value, raw: view.subarray(start, position), payload: view.subarray(payloadStart, position) });
  }
  return list;
}

function concatBytes(parts) {
  const out = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

const protoText = new TextEncoder();
const protoDecoder = new TextDecoder();
const bytesField = (field, payload) => concatBytes([Uint8Array.from([...writeVarint(field * 8 + 2), ...writeVarint(payload.length)]), payload]);
const varintField = (field, value) => Uint8Array.from([...writeVarint(field * 8), ...writeVarint(value)]);
const textField = (field, value) => bytesField(field, protoText.encode(value));

function intAttribute(name, value) {
  return concatBytes([textField(1, name), varintField(3, value), varintField(20, 2)]);
}

function intsAttribute(name, values) {
  return concatBytes([textField(1, name), ...values.map((value) => varintField(8, value)), varintField(20, 7)]);
}

function onnxNode(opType, inputs, outputs, name, attributes = []) {
  return concatBytes([
    ...inputs.map((input) => textField(1, input)),
    ...outputs.map((output) => textField(2, output)),
    textField(3, name),
    textField(4, opType),
    ...attributes.map((attribute) => bytesField(5, attribute)),
  ]);
}

function tensorDims(tensor) {
  const dims = [];
  for (const part of protoFields(tensor).filter((item) => item.field === 1)) {
    if (part.wire === 0) {
      dims.push(part.value);
    } else {
      let position = 0;
      while (position < part.payload.length) {
        let value;
        [value, position] = readVarint(part.payload, position);
        dims.push(value);
      }
    }
  }
  return dims;
}

// Standard-op equivalent of a 4-bit GatherBlockQuantized with gather_axis 0
// and quantize_axis 1: gather the packed rows, split each byte into its low
// and high nibble with integer Div/Mul/Sub, subtract the zero points and
// multiply by the per-block scales.
function gatherBlockQuantizedNodes({ data, ids, scales, zeroPoints, output, prefix, blocks, blockSize }) {
  const name = (suffix) => `${prefix}/${suffix}`;
  const nodes = [];
  const add = (opType, inputs, outputs, attributes) => nodes.push(onnxNode(opType, inputs, outputs, name(`${opType}_${nodes.length}`), attributes));
  add("Constant", [], [name("c16")], [intAttribute("value_int", 16)]);
  add("Constant", [], [name("last_axis")], [intsAttribute("value_ints", [-1])]);
  add("Constant", [], [name("value_shape")], [intsAttribute("value_ints", [0, 0, blocks, blockSize])]);
  add("Constant", [], [name("zero_shape")], [intsAttribute("value_ints", [0, 0, blocks, 1])]);
  add("Constant", [], [name("output_shape")], [intsAttribute("value_ints", [0, 0, blocks * blockSize])]);
  const unpack = (source, tag) => {
    add("Gather", [source, ids], [name(`${tag}_rows`)], [intAttribute("axis", 0)]);
    add("Cast", [name(`${tag}_rows`)], [name(`${tag}_int`)], [intAttribute("to", 7)]);
    add("Div", [name(`${tag}_int`), name("c16")], [name(`${tag}_high`)]);
    add("Mul", [name(`${tag}_high`), name("c16")], [name(`${tag}_high16`)]);
    add("Sub", [name(`${tag}_int`), name(`${tag}_high16`)], [name(`${tag}_low`)]);
    add("Unsqueeze", [name(`${tag}_low`), name("last_axis")], [name(`${tag}_low1`)]);
    add("Unsqueeze", [name(`${tag}_high`), name("last_axis")], [name(`${tag}_high1`)]);
    add("Concat", [name(`${tag}_low1`), name(`${tag}_high1`)], [name(`${tag}_pairs`)], [intAttribute("axis", -1)]);
    return name(`${tag}_pairs`);
  };
  add("Reshape", [unpack(data, "q"), name("value_shape")], [name("q_blocks")]);
  add("Cast", [name("q_blocks")], [name("q_float")], [intAttribute("to", 1)]);
  add("Reshape", [unpack(zeroPoints, "zp"), name("zero_shape")], [name("zp_blocks")]);
  add("Cast", [name("zp_blocks")], [name("zp_float")], [intAttribute("to", 1)]);
  add("Gather", [scales, ids], [name("scale_rows")], [intAttribute("axis", 0)]);
  add("Unsqueeze", [name("scale_rows"), name("last_axis")], [name("scale_blocks")]);
  add("Sub", [name("q_float"), name("zp_float")], [name("centered")]);
  add("Mul", [name("centered"), name("scale_blocks")], [name("scaled")]);
  add("Reshape", [name("scaled"), name("output_shape")], [output]);
  return nodes;
}

// ORT-web's WASM backend has no GatherBlockQuantized kernel, which every
// quantized EmbeddingGemma 2 export uses for its token embeddings. This
// rewrites such nodes into standard ops (identical results; checked against
// ORT's own kernel at cosine 1.000000) and copies everything else byte for
// byte. Only ModelProto.graph -> GraphProto.node is re-encoded. Nodes with
// another layout are left unchanged; `replaced` counts the rewritten ones.
export function rewriteGatherBlockQuantized(buffer) {
  const model = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  let replaced = 0;
  const rewriteGraph = (graph) => {
    const parts = protoFields(graph);
    const initializerDims = new Map();
    for (const part of parts.filter((item) => item.field === 5)) {
      const nameField = protoFields(part.payload).find((item) => item.field === 8);
      if (nameField) {
        initializerDims.set(protoDecoder.decode(nameField.payload), tensorDims(part.payload));
      }
    }
    const out = [];
    for (const part of parts) {
      const node = part.field === 1 ? protoFields(part.payload) : null;
      const opType = node?.find((item) => item.field === 4);
      if (!opType || protoDecoder.decode(opType.payload) !== "GatherBlockQuantized") {
        out.push(part.raw);
        continue;
      }
      const inputs = node.filter((item) => item.field === 1).map((item) => protoDecoder.decode(item.payload));
      const outputs = node.filter((item) => item.field === 2).map((item) => protoDecoder.decode(item.payload));
      const attributes = Object.fromEntries(node.filter((item) => item.field === 5).map((item) => {
        const fields = protoFields(item.payload);
        return [protoDecoder.decode(fields.find((f) => f.field === 1).payload), fields.find((f) => f.field === 3)?.value];
      }));
      const nodeName = protoDecoder.decode(node.find((item) => item.field === 3)?.payload ?? new Uint8Array());
      const [data, ids, scales, zeroPoints] = inputs;
      const dataDims = initializerDims.get(data);
      const scaleDims = initializerDims.get(scales);
      const blockSize = attributes.block_size;
      const supported = attributes.bits === 4 && (attributes.gather_axis ?? 0) === 0 && attributes.quantize_axis === 1
        && Boolean(zeroPoints) && dataDims?.length === 2 && scaleDims?.length === 2 && dataDims[1] * 2 === scaleDims[1] * blockSize;
      if (!supported) {
        out.push(part.raw);
        continue;
      }
      replaced += 1;
      for (const replacement of gatherBlockQuantizedNodes({ data, ids, scales, zeroPoints, output: outputs[0], prefix: `${nodeName}_std`, blocks: scaleDims[1], blockSize })) {
        out.push(bytesField(1, replacement));
      }
    }
    return concatBytes(out);
  };
  const out = protoFields(model).map((part) => (part.field === 7 && part.wire === 2 ? bytesField(7, rewriteGraph(part.payload)) : part.raw));
  return { bytes: concatBytes(out), replaced };
}
