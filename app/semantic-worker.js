// Web Worker for opt-in semantic search. It loads only the text half of the
// pinned EmbeddingGemma 2 ONNX model and turns query text into a normalized
// vector. Queries never leave the browser; only model files are downloaded.
import { chooseQueryBackend, readEncodedBody, rewriteGatherBlockQuantized } from "./semantic-search.js";

// Decoded size of ort-wasm-simd-threaded.asyncify.wasm for the onnxruntime-web
// build that transformers.js 4.3.2 pins; only used to scale progress.
const runtimeDecodedBytes = 26_861_777;
const transformersUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const modelHost = "https://huggingface.co";
// transformers.js keeps model files and the ORT WASM binary in this cache,
// keyed by URL; the worker stores config.json there too.
const runtimeCacheName = "transformers-cache";

let model = null;
let tokenizer = null;
let settings = null;
let loadPromise = null;
// Milliseconds since the load message, per phase; reported with "ready" so a
// slow device shows whether downloads, session creation or warmup dominate.
const timings = {};
let loadStartedAt = 0;
function mark(phase) {
  timings[phase] = Math.round(performance.now() - loadStartedAt);
}

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function fetchTextConfig({ modelId, revision }) {
  const url = `${modelHost}/${modelId}/resolve/${encodeURIComponent(revision)}/config.json`;
  // The URL is revision-pinned, so a cached copy never goes stale; reading
  // it from the cache keeps a restore from waiting on the network.
  const cache = await caches.open(runtimeCacheName).catch(() => null);
  let response = await cache?.match(url);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) {
      throw new Error(`無法讀取模型設定：HTTP ${response.status}`);
    }
    await cache?.put(url, response.clone()).catch(() => {});
  }
  const config = await response.json();
  // Removing the vision and audio sub-configs keeps transformers.js from
  // downloading those encoders; the query side only needs text.
  delete config.vision_config;
  delete config.audio_config;
  return config;
}

// "local" once tokenizer.json came from this site, "hub" after falling back.
// Unless it fell back, its progress is reported here in transfer bytes, not by
// transformers.js in decoded bytes (also when it is restored from the cache).
let tokenizerSource = "";

function progressCallback(info) {
  if (info?.file === "tokenizer.json" && settings?.tokenizer && tokenizerSource !== "hub") {
    return;
  }
  if (info?.status === "progress" && info.file) {
    post({ type: "progress", file: info.file, loaded: Number(info.loaded) || 0, total: Number(info.total) || 0 });
  }
}

// Once every model file is downloaded, ORT builds the inference session, which
// reports no byte progress; tell the page so it can show that stage.
function modelProgressCallback() {
  const files = new Map();
  let announced = false;
  return (info) => {
    progressCallback(info);
    if (!info?.file) {
      return;
    }
    if (info.status === "initiate") {
      files.set(info.file, false);
    } else if (info.status === "done" && files.has(info.file)) {
      files.set(info.file, true);
      if (!announced && [...files.values()].every(Boolean)) {
        announced = true;
        mark("modelFiles");
        post({ type: "stage", stage: "initializing" });
      }
    }
  };
}

async function cachedRuntime(url) {
  try {
    const cache = await caches.open(runtimeCacheName);
    return (await cache.match(url)) ?? null;
  } catch {
    return null;
  }
}

// ORT only fetches its WASM binary when the first session is created, i.e.
// after the whole model has downloaded. Fetch it up front instead, in
// parallel with the model files, and report its progress. jsDelivr sends it
// compressed, so progress counts decoded bytes against the decoded size.
// Returns null when transformers.js should load the binary itself (already
// cached, or the prefetch failed and ORT's own fetch is the fallback).
async function prefetchRuntime(url) {
  try {
    return await downloadRuntime(url);
  } catch {
    return null;
  }
}

async function downloadRuntime(url) {
  if (await cachedRuntime(url)) {
    post({ type: "runtime-progress", loaded: runtimeDecodedBytes, total: runtimeDecodedBytes });
    return null;
  }
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`無法下載執行環境：HTTP ${response.status}`);
  }
  // Content-Encoding is not exposed cross-origin, and a compressed response
  // declares its compressed length, so never trust a smaller declared size.
  const total = Math.max(Number(response.headers.get("content-length")) || 0, runtimeDecodedBytes);
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    chunks.push(value);
    loaded += value.byteLength;
    post({ type: "runtime-progress", loaded: Math.min(loaded, total), total });
  }
  post({ type: "runtime-progress", loaded, total: loaded });
  return new Blob(chunks).arrayBuffer();
}

// Debug mode only (`?semantic-debug`): forward ORT warnings and, with
// `semantic-placement`, the node-placement summary from its verbose log, so a
// slow device shows whether operators fell back from WebGPU to the CPU. ORT
// writes every verbose line through console.error, hence the block tracking.
function installDebugCapture() {
  let forwarded = 0;
  let placementLines = 0;
  for (const level of ["warn", "error", "log", "info", "debug"]) {
    const original = console[level].bind(console);
    console[level] = (...args) => {
      const text = args.map((arg) => (typeof arg === "string" ? arg : errorMessage(arg))).join(" ");
      if (settings.placement && /Node placements/.test(text)) {
        placementLines = 80;
      }
      const isWarning = /\[(W|E):onnxruntime/.test(text) || (!settings.placement && (level === "warn" || level === "error"));
      const wanted = placementLines > 0 || isWarning;
      if (placementLines > 0) {
        placementLines -= 1;
      }
      if (wanted && forwarded < 120) {
        forwarded += 1;
        post({ type: "log", level, text: text.slice(0, 400) });
      }
      if (!settings.placement || wanted) {
        original(...args);
      }
    };
  }
}

// The WASM backend cannot run GatherBlockQuantized, so before the session is
// created the model graph (about 0.5 MB, without weights) is rewritten into
// standard ops and stored under its own URL, where transformers.js reads it.
// The rewrite is idempotent, so a graph cached by an earlier visit is fine.
async function prepareWasmGraph({ modelId, revision, dtype }) {
  const file = dtype === "fp32" ? "model.onnx" : `model_${dtype}.onnx`;
  const url = `${modelHost}/${modelId}/resolve/${encodeURIComponent(revision)}/onnx/${file}`;
  const cache = await caches.open(runtimeCacheName);
  let response = await cache.match(url);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) {
      throw new Error(`無法讀取模型結構：HTTP ${response.status}`);
    }
  }
  const { bytes, replaced } = rewriteGatherBlockQuantized(await response.arrayBuffer());
  if (replaced > 0 || !(await cache.match(url))) {
    await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream", "content-length": String(bytes.length) } }));
  }
}

// huggingface.co sends tokenizer.json (32 MB) uncompressed; this site ships
// the same file gzipped (about 5.4 MB). The decoded bytes must match the
// pinned file's sha256, otherwise the caller falls back to huggingface.co.
async function fetchSiteTokenizer(tokenizer) {
  const response = await fetch(tokenizer.url);
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  const bytes = await readEncodedBody(response, {
    expectedBytes: tokenizer.bytes,
    onProgress: (loaded) => post({ type: "progress", file: "tokenizer.json", loaded: Math.min(loaded, tokenizer.transferBytes), total: tokenizer.transferBytes }),
  });
  const digest = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (byte) => byte.toString(16).padStart(2, "0")).join("");
  if (digest !== tokenizer.sha256) {
    throw new Error("tokenizer sha256 mismatch");
  }
  return new Response(bytes, { headers: { "content-type": "application/json; charset=utf-8" } });
}

async function detectAdapter() {
  try {
    return (await navigator.gpu?.requestAdapter()) ?? null;
  } catch {
    return null;
  }
}

async function loadModel(message) {
  loadStartedAt = performance.now();
  settings = {
    modelId: message.modelId,
    revision: message.revision,
    dtype: message.dtype || "q4",
    debug: Boolean(message.debug),
    placement: Boolean(message.placement),
    backend: String(message.backend ?? ""),
    tokenizer: message.tokenizer?.url ? message.tokenizer : null,
    dims: Number(message.dims) || 512,
    queryPrefix: String(message.queryPrefix ?? ""),
  };
  const adapter = await detectAdapter();
  const info = adapter?.info;
  const backend = chooseQueryBackend({ hasWebGpu: Boolean(adapter), vendor: info?.vendor ?? "", override: settings.backend });
  timings.backend = backend;
  timings.gpu = [info?.vendor, info?.architecture, info?.device].filter(Boolean).join("/") || "none";
  if (settings.debug) {
    timings.features = adapter ? ["shader-f16", "subgroups", "timestamp-query"].filter((name) => adapter.features.has(name)).join("+") || "none" : "none";
    timings.dtype = settings.dtype;
    timings.isolated = String(self.crossOriginIsolated === true);
  }
  const configPromise = fetchTextConfig(settings);
  const { AutoModel, AutoTokenizer, env } = await import(transformersUrl);
  mark("import");
  if (settings.debug) {
    installDebugCapture();
  }
  // Pin every Hub request, including transformers.js internal probes that do
  // not forward `revision`, to the fixed model revision.
  env.remotePathTemplate = `{model}/resolve/${encodeURIComponent(settings.revision)}/`;

  const runtimeUrl = env.backends.onnx.wasm?.wasmPaths?.wasm;
  const runtimePromise = (runtimeUrl ? prefetchRuntime(runtimeUrl) : Promise.resolve(null)).finally(() => mark("runtime"));
  // Hand the prefetched binary to transformers.js when ORT asks for it, so it
  // is downloaded once; transformers.js then stores it in its cache as usual.
  const baseFetch = env.fetch;
  const hubTokenizerUrl = `${modelHost}/${settings.modelId}/resolve/${encodeURIComponent(settings.revision)}/tokenizer.json`;
  env.fetch = async (url, init) => {
    if (settings.tokenizer && String(url) === hubTokenizerUrl && tokenizerSource !== "hub" && new Headers(init?.headers).has("range")) {
      // transformers.js probes the file with a range request first; answer it
      // here so huggingface.co is not contacted for the tokenizer at all.
      return new Response(new Uint8Array(1), {
        status: 206,
        headers: { "content-range": `bytes 0-0/${settings.tokenizer.bytes}`, "content-length": "1", "content-type": "application/json" },
      });
    }
    if (settings.tokenizer && String(url) === hubTokenizerUrl && tokenizerSource !== "hub") {
      try {
        const response = await fetchSiteTokenizer(settings.tokenizer);
        tokenizerSource = "local";
        return response;
      } catch {
        tokenizerSource = "hub";
      }
    }
    if (runtimeUrl && String(url) === runtimeUrl) {
      const buffer = await runtimePromise;
      if (buffer) {
        return new Response(buffer, { headers: { "content-type": "application/wasm" } });
      }
    }
    return baseFetch(url, init);
  };

  const graphPromise = backend === "wasm" ? prepareWasmGraph(settings) : Promise.resolve();

  const tokenizerPromise = AutoTokenizer.from_pretrained(settings.modelId, {
    revision: settings.revision,
    progress_callback: progressCallback,
  }).finally(() => mark("tokenizer"));
  const modelPromise = Promise.all([configPromise.finally(() => mark("config")), graphPromise]).then(([config]) =>
    AutoModel.from_pretrained(settings.modelId, {
      config,
      dtype: settings.dtype,
      revision: settings.revision,
      device: backend,
      progress_callback: modelProgressCallback(),
      ...(settings.placement ? { session_options: { logSeverityLevel: 0, logVerbosityLevel: 1 } } : {}),
    }),
  ).finally(() => mark("session"));
  [tokenizer, model] = await Promise.all([tokenizerPromise, modelPromise, runtimePromise]);
  // A first embedding compiles the GPU shaders (on WebGPU) and confirms the
  // backend can actually run the graph.
  await embedText("warmup");
  mark("warmup");
  timings.tokenizer = tokenizerSource || "cache";
  return backend;
}

async function embedText(text) {
  const inputs = await tokenizer([`${settings.queryPrefix}${text}`], { padding: true, truncation: true });
  const output = await model(inputs);
  const data = output.sentence_embedding.data;
  const length = Math.min(settings.dims, data.length);
  const vector = new Float32Array(length);
  let sum = 0;
  for (let index = 0; index < length; index += 1) {
    vector[index] = data[index];
    sum += data[index] * data[index];
  }
  const inverse = sum > 0 ? 1 / Math.sqrt(sum) : 0;
  for (let index = 0; index < length; index += 1) {
    vector[index] *= inverse;
  }
  return vector;
}

self.addEventListener("message", async (event) => {
  const message = event.data ?? {};
  if (message.type === "load") {
    loadPromise ??= loadModel(message);
    try {
      const device = await loadPromise;
      post({ type: "ready", device, timings: { ...timings } });
    } catch (error) {
      loadPromise = null;
      post({ type: "error", message: errorMessage(error) });
    }
    return;
  }
  if (message.type === "embed") {
    try {
      if (!loadPromise) {
        throw new Error("查詢模型尚未載入");
      }
      await loadPromise;
      const startedAt = performance.now();
      const vector = await embedText(String(message.text ?? ""));
      const ms = Math.round(performance.now() - startedAt);
      post({ type: "embedding", requestId: message.requestId, vector, ms }, [vector.buffer]);
    } catch (error) {
      post({ type: "error", requestId: message.requestId, message: errorMessage(error) });
    }
  }
});
