import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  isModelCacheUrl,
  modelFileUrls,
  readSemanticPreference,
  writeSemanticPreference,
} from "../app/semantic-controller.js";
import { matchesFilters } from "../app/search-sort.js";
import {
  chooseQueryBackend,
  eligibilityMask,
  enterSimilarState,
  exitSimilarState,
  fuseRankings,
  hasVector,
  normalizeVector,
  parseVectorMeta,
  parseVectorSet,
  rankScores,
  rewriteGatherBlockQuantized,
  resolveResultView,
  rowVector,
  scoreRows,
  semanticRanking,
  similarRanking,
  splitResultTabs,
  topRows,
} from "../app/semantic-search.js";
import { decodeUrlState, encodeUrlState } from "../app/url-state.js";

const dims = 4;
const rowBytes = dims + 4;

// Builds a quantized vector file the same way the build does: int8 values per
// row followed by a float32 little-endian scale; null rows stay all zero.
function quantizedBuffer(rows) {
  const buffer = new ArrayBuffer(rows.length * rowBytes);
  const values = new Int8Array(buffer);
  const view = new DataView(buffer);
  rows.forEach((row, index) => {
    if (!row) {
      return;
    }
    const normalized = normalizeVector(Float32Array.from(row));
    const scale = Math.max(...normalized.map(Math.abs)) / 127;
    for (let dim = 0; dim < dims; dim += 1) {
      values[index * rowBytes + dim] = Math.max(-127, Math.min(127, Math.round(normalized[dim] / scale)));
    }
    view.setFloat32(index * rowBytes + dims, scale, true);
  });
  return buffer;
}

function vectorSet(rows) {
  return parseVectorSet(quantizedBuffer(rows), { dims, rowBytes, rowCount: rows.length });
}

function meta(overrides = {}) {
  return {
    formatVersion: "2026-11-embeddings-v1",
    dims: 512,
    rowBytes: 516,
    rowCount: 3,
    queryPrefix: "task: search result | query: ",
    sourceModel: "EmbeddingGemma 2 (8-bit, 768d truncated to 512d)",
    queryModel: { id: "onnx-community/embeddinggemma-2-ONNX", revision: "abc", dtype: "q4" },
    photosSha256: "f".repeat(64),
    sets: {
      text: { file: "text-int8.bin", bytes: 1548, sha256: "a".repeat(64), covered: 2 },
      image: { file: "image-int8.bin", bytes: 1548, sha256: "b".repeat(64), covered: 3 },
    },
    ...overrides,
  };
}

describe("semantic vector parsing", () => {
  it("parses meta and rejects inconsistent sizes", () => {
    const parsed = parseVectorMeta(meta());
    assert.equal(parsed.rowCount, 3);
    assert.equal(parsed.sets.image.file, "image-int8.bin");
    assert.equal(parsed.queryModel.dtype, "q4");
    assert.equal(parsed.sourceModel, "EmbeddingGemma 2 (8-bit, 768d truncated to 512d)");
    assert.equal(parseVectorMeta(meta({ sourceModel: undefined })).sourceModel, "");
    assert.throws(() => parseVectorMeta(meta({ rowBytes: 512 })), /每列大小/);
    assert.throws(() => parseVectorMeta(meta({ formatVersion: "other" })), /不支援/);
    assert.throws(
      () => parseVectorMeta(meta({ sets: { image: { file: "image-int8.bin", bytes: 10, sha256: "b".repeat(64) } } })),
      /image 向量檔/,
    );
  });

  it("reads int8 rows with float32 scales and treats zero scale as missing", () => {
    const set = vectorSet([[1, 0, 0, 0], null, [0.5, 0.5, 0.5, 0.5]]);
    assert.equal(set.covered, 2);
    assert.equal(hasVector(set, 0), true);
    assert.equal(hasVector(set, 1), false);
    assert.equal(rowVector(set, 1), null);
    const restored = rowVector(set, 2);
    for (const value of restored) {
      assert.ok(Math.abs(value - 0.5) < 1e-6);
    }
    assert.throws(() => parseVectorSet(new ArrayBuffer(rowBytes), { dims, rowBytes, rowCount: 2 }), /大小不符/);
  });

  it("accepts Uint8Array views at a byte offset", () => {
    const source = new Uint8Array(quantizedBuffer([[0, 1, 0, 0]]));
    const padded = new Uint8Array(source.length + 3);
    padded.set(source, 3);
    const set = parseVectorSet(padded.subarray(3), { dims, rowBytes, rowCount: 1 });
    assert.equal(set.covered, 1);
    assert.ok(Math.abs(scoreRows(set, Float32Array.from([0, 1, 0, 0]))[0] - 1) < 1e-6);
  });
});

describe("semantic ranking", () => {
  const rows = [
    [1, 0, 0, 0],
    [0.9, 0.1, 0, 0],
    [0, 1, 0, 0],
    null,
    [-1, 0, 0, 0],
  ];
  const query = normalizeVector(Float32Array.from([1, 0, 0, 0]));

  it("scores cosine similarity and ranks top-K, skipping missing vectors", () => {
    const set = vectorSet(rows);
    const scores = scoreRows(set, query);
    assert.ok(Math.abs(scores[0] - 1) < 1e-6);
    assert.ok(Math.abs(scores[4] + 1) < 1e-6);
    assert.equal(scores[3], Number.NEGATIVE_INFINITY);
    assert.deepEqual(topRows(scores, 3), [0, 1, 2]);
    assert.deepEqual(topRows(scores, 10), [0, 1, 2, 4]);
  });

  it("breaks score ties by row order", () => {
    const scores = Float32Array.from([0.5, 0.5, 0.9, 0.5]);
    const output = new Uint32Array(4);
    const count = rankScores(scores, { output });
    assert.deepEqual(Array.from(output.subarray(0, count)), [2, 0, 1, 3]);
  });

  it("fuses rankings with reciprocal rank fusion", () => {
    // Row 1 is second in both lists: 2/62 beats 1/61 + 1/64 for rows that
    // top one list and finish last in the other. Ties keep row order.
    assert.deepEqual(fuseRankings([[0, 1, 2, 3], [3, 1, 2, 0]], { rowCount: 4, k: 60 }), [1, 0, 3, 2]);
    // A row present in only one list still ranks, below rows in both.
    assert.deepEqual(fuseRankings([[0, 1], [1]], { rowCount: 3, k: 60 }), [1, 0]);
    assert.deepEqual(fuseRankings([[0, 1, 2]], { rowCount: 3, k: 60, limit: 2 }), [0, 1]);
  });

  it("applies other filters before taking the semantic top N", () => {
    const photos = [
      { photo_id: "1", orientation: "landscape", search_text: "" },
      { photo_id: "2", orientation: "portrait", search_text: "" },
      { photo_id: "3", orientation: "landscape", search_text: "" },
      { photo_id: "4", orientation: "landscape", search_text: "" },
      { photo_id: "5", orientation: "landscape", search_text: "" },
    ];
    const eligible = eligibilityMask(photos, (photo) => matchesFilters(photo, { search: "", orientation: ["landscape"] }));
    const text = vectorSet(rows);
    const result = semanticRanking({ sets: { text }, query, eligible, topK: 2, rrfK: 60 });
    assert.deepEqual(result, [0, 2]);
  });

  it("fuses text and image rankings and keeps rows that have any vector", () => {
    const text = vectorSet([[1, 0, 0, 0], null, [0, 1, 0, 0]]);
    const image = vectorSet([[0, 1, 0, 0], [1, 0, 0, 0], null]);
    const result = semanticRanking({ sets: { text, image }, query, topK: 10, rrfK: 60 });
    assert.deepEqual(new Set(result), new Set([0, 1, 2]));
    // Row 0 appears in both lists and ranks first after fusion.
    assert.equal(result[0], 0);
  });

  it("never returns photos without any vector", () => {
    const text = vectorSet([[1, 0, 0, 0], null, [0, 1, 0, 0]]);
    const image = vectorSet([[1, 0, 0, 0], null, null]);
    const result = semanticRanking({ sets: { text, image }, query, topK: 10, rrfK: 60 });
    assert.equal(result.includes(1), false);
    assert.deepEqual(semanticRanking({ sets: {}, query, topK: 10 }), []);
  });

  it("ranks similar photos by image vectors and excludes the source photo", () => {
    const image = vectorSet(rows);
    assert.deepEqual(similarRanking({ image, sourceRow: 0, topK: 3 }), [1, 2, 4]);
    const eligible = Uint8Array.from([1, 0, 1, 1, 1]);
    assert.deepEqual(similarRanking({ image, sourceRow: 0, eligible, topK: 3 }), [2, 4]);
    assert.deepEqual(similarRanking({ image, sourceRow: 3, topK: 3 }), []);
  });

  it("reuses scratch buffers across repeated rankings", () => {
    const set = vectorSet(rows);
    const first = semanticRanking({ sets: { image: set }, query, topK: 5 });
    const second = semanticRanking({ sets: { image: set }, query: normalizeVector(Float32Array.from([0, 1, 0, 0])), topK: 5 });
    assert.deepEqual(first, [0, 1, 2, 4]);
    assert.deepEqual(second.slice(0, 2), [2, 1]);
    assert.deepEqual(semanticRanking({ sets: { image: set }, query, topK: 5 }), first);
  });
});

describe("semantic result views", () => {
  it("shows tabs only for ready text search with a query", () => {
    assert.deepEqual(resolveResultView({ textReady: true, query: "人很多" }), { mode: "semantic", showTabs: true });
    assert.deepEqual(resolveResultView({ textReady: true, query: "人很多", requestedView: "keyword" }), {
      mode: "keyword",
      showTabs: true,
    });
    assert.deepEqual(resolveResultView({ textReady: true, query: "  " }), { mode: "keyword", showTabs: false });
    assert.deepEqual(resolveResultView({ textReady: false, query: "人很多", requestedView: "semantic" }), {
      mode: "keyword",
      showTabs: false,
    });
  });

  it("uses similar view only when image vectors are ready", () => {
    assert.deepEqual(resolveResultView({ imageReady: true, similarPhotoId: "1" }), { mode: "similar", showTabs: false });
    assert.deepEqual(resolveResultView({ imageReady: false, similarPhotoId: "1" }), { mode: "keyword", showTabs: false });
  });

  it("splits keyword and semantic tabs without changing keyword results", () => {
    const photos = [{ photo_id: "1" }, { photo_id: "2" }, { photo_id: "3" }];
    const keywordResults = [photos[2]];
    const tabs = splitResultTabs({ photos, keywordResults, semanticRows: [1, 0] });
    assert.equal(tabs.keyword, keywordResults);
    assert.deepEqual(tabs.semantic.map((photo) => photo.photo_id), ["2", "1"]);
  });
});

describe("semantic preferences and URL state", () => {
  it("stores only full or image modes and tolerates blocked storage", () => {
    const store = new Map();
    const storage = {
      getItem: (key) => store.get(key) ?? null,
      setItem: (key, value) => store.set(key, value),
      removeItem: (key) => store.delete(key),
    };
    writeSemanticPreference("full", storage);
    assert.equal(readSemanticPreference(storage), "full");
    writeSemanticPreference("", storage);
    assert.equal(readSemanticPreference(storage), "");
    store.set("coscup-photo-finder-semantic", "other");
    assert.equal(readSemanticPreference(storage), "");
    const blocked = {
      getItem() {
        throw new Error("blocked");
      },
      setItem() {
        throw new Error("blocked");
      },
    };
    assert.equal(readSemanticPreference(blocked), "");
    assert.doesNotThrow(() => writeSemanticPreference("image", blocked));
  });

  it("checks the pinned model files in transformers-cache", () => {
    assert.deepEqual(modelFileUrls({ id: "onnx-community/embeddinggemma-2-ONNX", revision: "abc" }), [
      "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/abc/onnx/model_q4.onnx",
      "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/abc/onnx/model_q4.onnx_data",
      "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/abc/tokenizer.json",
      "https://huggingface.co/onnx-community/embeddinggemma-2-ONNX/resolve/abc/tokenizer_config.json",
    ]);
  });

  it("round-trips view and similar URL parameters", () => {
    const params = encodeUrlState({ search: "擺攤", view: "keyword", similar: "55244854377", filters: {} });
    assert.equal(params.get("view"), "keyword");
    assert.equal(params.get("similar"), "55244854377");
    const decoded = decodeUrlState(params);
    assert.equal(decoded.view, "keyword");
    assert.equal(decoded.similar, "55244854377");
    assert.equal(decoded.search, "擺攤");
    assert.equal(encodeUrlState({ view: "other", filters: {} }).get("view"), null);
    assert.equal(decodeUrlState(new URLSearchParams("view=other")).view, "");
    assert.equal(encodeUrlState({ filters: {} }).toString(), "");
  });
});

describe("semantic cache and similar-state transitions", () => {
  it("matches only this model's transformers-cache entries", () => {
    const id = "onnx-community/embeddinggemma-2-ONNX";
    assert.equal(isModelCacheUrl(`https://huggingface.co/${id}/resolve/abc/onnx/model_q4.onnx`, id), true);
    assert.equal(isModelCacheUrl(`https://huggingface.co/${id}-other/resolve/abc/config.json`, id), false);
    assert.equal(isModelCacheUrl("https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.wasm", id), false);
    assert.equal(isModelCacheUrl(`https://huggingface.co/${id}/resolve/abc/config.json`, ""), false);
  });

  it("remembers the query and tab replaced by a similar search and restores them", () => {
    const before = { search: "擺攤", view: "keyword", similar: "", similarReturn: null };
    const entered = enterSimilarState(before, "100");
    assert.deepEqual(entered, { search: "", view: "keyword", similar: "100", similarReturn: { search: "擺攤", view: "keyword" } });
    // Chaining to another photo keeps the original query to return to.
    const chained = enterSimilarState(entered, "200");
    assert.equal(chained.similar, "200");
    assert.deepEqual(chained.similarReturn, { search: "擺攤", view: "keyword" });
    assert.deepEqual(exitSimilarState(chained), { search: "擺攤", view: "keyword", similar: "", similarReturn: null });
  });

  it("clears a similar condition from a shared link without a remembered query", () => {
    assert.deepEqual(exitSimilarState({ search: "", view: "", similar: "100", similarReturn: null }), {
      search: "",
      view: "",
      similar: "",
      similarReturn: null,
    });
  });
});

describe("query model backend", () => {
  const varint = (value) => {
    const out = [];
    while (value >= 0x80) {
      out.push((value % 128) | 0x80);
      value = Math.floor(value / 128);
    }
    out.push(value);
    return out;
  };
  const bytesField = (field, payload) => [...varint(field * 8 + 2), ...varint(payload.length), ...payload];
  const text = (value) => [...new TextEncoder().encode(value)];
  const intAttr = (name, value) => bytesField(5, [...bytesField(1, text(name)), ...varint(3 * 8), ...varint(value), ...varint(20 * 8), 2]);
  const initializer = (name, dims, type) => bytesField(5, [...dims.flatMap((d) => [...varint(1 * 8), ...varint(d)]), ...varint(2 * 8), type, ...bytesField(8, text(name))]);
  const gbqModel = (bits = 4) => {
    const gbq = bytesField(1, [
      ...["table", "input_ids", "scales", "zero_points"].flatMap((input) => bytesField(1, text(input))),
      ...bytesField(2, text("embeds")),
      ...bytesField(3, text("/embed/Gather_Quant")),
      ...bytesField(4, text("GatherBlockQuantized")),
      ...intAttr("bits", bits),
      ...intAttr("block_size", 4),
      ...intAttr("gather_axis", 0),
      ...intAttr("quantize_axis", 1),
      ...bytesField(7, text("com.microsoft")),
    ]);
    const relu = bytesField(1, [...bytesField(1, text("embeds")), ...bytesField(2, text("out")), ...bytesField(3, text("/relu")), ...bytesField(4, text("Relu"))]);
    const graph = [...gbq, ...relu, ...bytesField(2, text("main")), ...initializer("table", [8, 2], 2), ...initializer("scales", [8, 1], 1), ...initializer("zero_points", [8, 1], 2)];
    return Uint8Array.from([...varint(1 * 8), 9, ...bytesField(7, graph), ...bytesField(8, text("opset"))]);
  };
  // Reads node op types and outputs back out of the rewritten model.
  const nodes = (bytes) => {
    const read = (view, position) => {
      let value = 0;
      let shift = 0;
      for (;;) {
        const byte = view[position++];
        value += (byte & 0x7f) * 2 ** shift;
        if (byte < 0x80) return [value, position];
        shift += 7;
      }
    };
    const fields = (view) => {
      const out = [];
      let position = 0;
      while (position < view.length) {
        let key;
        [key, position] = read(view, position);
        if (key % 8 === 0) {
          [, position] = read(view, position);
          continue;
        }
        let length;
        [length, position] = read(view, position);
        out.push([Math.floor(key / 8), view.subarray(position, position + length)]);
        position += length;
      }
      return out;
    };
    const decoder = new TextDecoder();
    const graph = fields(bytes).find(([field]) => field === 7)[1];
    return fields(graph).filter(([field]) => field === 1).map(([, node]) => {
      const parts = fields(node);
      return { op: decoder.decode(parts.find(([f]) => f === 4)[1]), outputs: parts.filter(([f]) => f === 2).map(([, v]) => decoder.decode(v)) };
    });
  };

  it("prefers WebGPU except without an adapter or on slow GPU vendors", () => {
    assert.equal(chooseQueryBackend({ hasWebGpu: true, vendor: "apple" }), "webgpu");
    assert.equal(chooseQueryBackend({ hasWebGpu: true, vendor: "ARM" }), "wasm");
    assert.equal(chooseQueryBackend({ hasWebGpu: false }), "wasm");
    assert.equal(chooseQueryBackend({ hasWebGpu: true, vendor: "arm", override: "webgpu" }), "webgpu");
    assert.equal(chooseQueryBackend({ hasWebGpu: false, override: "webgpu" }), "wasm");
    assert.equal(chooseQueryBackend({ hasWebGpu: true, vendor: "apple", override: "wasm" }), "wasm");
  });

  it("rewrites 4-bit GatherBlockQuantized into standard ops and keeps its output", () => {
    const { bytes, replaced } = rewriteGatherBlockQuantized(gbqModel());
    assert.equal(replaced, 1);
    const ops = nodes(bytes);
    assert.equal(ops.some((node) => node.op === "GatherBlockQuantized"), false);
    assert.deepEqual(ops.at(-1), { op: "Relu", outputs: ["out"] });
    assert.deepEqual(ops.at(-2), { op: "Reshape", outputs: ["embeds"] });
    assert.ok(ops.every((node) => ["Constant", "Gather", "Cast", "Div", "Mul", "Sub", "Unsqueeze", "Concat", "Reshape", "Relu"].includes(node.op)));
    // Rewriting again is a no-op, so a cached rewritten graph stays valid.
    const again = rewriteGatherBlockQuantized(bytes.buffer);
    assert.equal(again.replaced, 0);
    assert.deepEqual(again.bytes, bytes);
  });

  it("leaves unsupported layouts and malformed input alone", () => {
    const model = gbqModel(8);
    const { bytes, replaced } = rewriteGatherBlockQuantized(model);
    assert.equal(replaced, 0);
    assert.deepEqual(bytes, model);
    assert.throws(() => rewriteGatherBlockQuantized(Uint8Array.from([0x80])), /格式不正確/);
  });
});
