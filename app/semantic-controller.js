// Opt-in semantic search lifecycle: consent dialog, downloads with progress,
// Cache API reuse, the query worker, and the per-device enabled preference.
// Ranking stays in semantic-search.js; this module owns network and storage.
import { parseVectorMeta, parseVectorSet, resolveSemanticMode } from "./semantic-search.js";

export const semanticStorageKey = "coscup-photo-finder-semantic";
// Must not start with "coscup-photo-finder-": the service worker deletes those
// caches on every deployment, while vector caches should survive unchanged.
export const vectorCachePrefix = "photo-finder-semantic-vectors-";
export const modelCacheName = "transformers-cache";
export const modelCacheFiles = ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json"];
const modelHost = "https://huggingface.co";
// Transfer sizes. The runtime (transformers.js plus the ORT WASM binary) comes
// compressed from jsDelivr: about 5.7 MB on the wire, 27 MB once decoded.
const estimatedBytes = { model: 206_000_000, runtime: 5_700_000, vectorSet: 17_715_312 };
const embeddingCacheSize = 32;

export const noWebGpuMessage = "這個瀏覽器不支援 WebGPU，無法執行文字語意搜尋；仍可使用以圖找圖。";

// Text search runs only on WebGPU (the quantized model needs an operator the
// WASM backend lacks), so check for a usable adapter before any download.
export async function detectWebGpu(nav = globalThis.navigator) {
  try {
    if (!nav?.gpu || typeof nav.gpu.requestAdapter !== "function") {
      return false;
    }
    return (await nav.gpu.requestAdapter()) !== null;
  } catch {
    return false;
  }
}

// transformers-cache keys are full Hub URLs; entries for this model share the
// `https://huggingface.co/<id>/` prefix whatever the file or revision.
export function isModelCacheUrl(url, modelId) {
  return Boolean(modelId) && String(url).startsWith(`${modelHost}/${modelId}/`);
}

export function readSemanticPreference(storage = globalThis.localStorage) {
  try {
    const value = storage?.getItem(semanticStorageKey);
    return value === "full" || value === "image" ? value : "";
  } catch {
    return "";
  }
}

export function writeSemanticPreference(mode, storage = globalThis.localStorage) {
  try {
    if (mode === "full" || mode === "image") {
      storage?.setItem(semanticStorageKey, mode);
    } else {
      storage?.removeItem(semanticStorageKey);
    }
  } catch {
    // The feature still works for this page when storage is blocked.
  }
}

// `?semantic-debug` shows query-model timings on the page and sends them to
// the serving host, so a slow phone can be diagnosed without remote DevTools.
// `semantic-placement` reports which execution provider ORT assigned nodes to,
// `semantic-dtype` tries another quantization of the same model and
// `semantic-matmul=cpu|gpu` forces where MatMulNBits runs; all need
// `semantic-debug`.
const debugParams = (() => {
  try {
    return new URLSearchParams(globalThis.location?.search ?? "");
  } catch {
    return new URLSearchParams();
  }
})();
const semanticDebug = debugParams.has("semantic-debug");

function reportTiming(kind, values) {
  console.info(`[semantic] ${kind}`, values);
  if (!semanticDebug || typeof document === "undefined") {
    return;
  }
  let panel = document.getElementById("semanticDebugPanel");
  if (!panel) {
    panel = document.createElement("pre");
    panel.id = "semanticDebugPanel";
    panel.style.cssText = "position:fixed;left:8px;bottom:8px;z-index:9999;max-width:calc(100vw - 16px);margin:0;padding:6px 8px;border-radius:6px;background:rgba(0,0,0,.8);color:#fff;font:11px/1.4 ui-monospace,monospace;white-space:pre-wrap;pointer-events:none";
    document.body.append(panel);
  }
  const line = `${kind} ${Object.entries(values).map(([key, value]) => `${key}=${value}`).join(" ")}`;
  panel.textContent = `${panel.textContent}${panel.textContent ? "\n" : ""}${line}`.split("\n").slice(-8).join("\n");
  fetch(`./semantic-debug?${new URLSearchParams({ kind, ua: navigator.userAgent, ...values })}`, { cache: "no-store" }).catch(() => {});
}

export function modelFileUrls({ id, revision }) {
  return modelCacheFiles.map((file) => `${modelHost}/${id}/resolve/${encodeURIComponent(revision)}/${file}`);
}

export function formatMegabytes(bytes) {
  return `${Math.max(1, Math.round(bytes / 1_000_000))} MB`;
}

// Full mode uses every vector set the deployment provides; image-only mode
// needs the image set.
export function setsForMode(mode, meta) {
  if (mode !== "full") {
    return ["image"];
  }
  return ["text", "image"].filter((kind) => !meta || meta.sets[kind]);
}

function hasCacheApi() {
  return typeof caches !== "undefined";
}

class CancelledError extends Error {
  constructor() {
    super("已取消下載");
    this.name = "CancelledError";
  }
}

function isCancelled(error) {
  return error instanceof CancelledError || error?.name === "AbortError";
}

async function sha256Hex(bytes) {
  if (!globalThis.crypto?.subtle) {
    return "";
  }
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (value) => value.toString(16).padStart(2, "0")).join("");
}

function queryDom() {
  const byId = (id) => document.getElementById(id);
  return {
    bar: byId("semanticBar"),
    button: byId("semanticButton"),
    status: byId("semanticStatus"),
    dialog: byId("semanticDialog"),
    message: byId("semanticDialogMessage"),
    webgpuNote: byId("semanticWebGpuNote"),
    choice: byId("semanticChoice"),
    sizeVectors: byId("semanticSizeVectors"),
    sizeVectorsLabel: byId("semanticSizeVectorsLabel"),
    sizeModelRow: byId("semanticSizeModelRow"),
    sizeRuntimeRow: byId("semanticSizeRuntimeRow"),
    sizeTotalRow: byId("semanticSizeTotalRow"),
    sizeModel: byId("semanticSizeModel"),
    sourceModel: byId("semanticSourceModel"),
    queryModelLink: byId("semanticQueryModelLink"),
    queryModelDetail: byId("semanticQueryModelDetail"),
    sizeRuntime: byId("semanticSizeRuntime"),
    sizeTotal: byId("semanticSizeTotal"),
    enableFull: byId("semanticEnableFullButton"),
    enableImage: byId("semanticEnableImageButton"),
    dismiss: byId("semanticDismissButton"),
    progress: byId("semanticProgress"),
    progressVectors: byId("semanticProgressVectors"),
    progressVectorsText: byId("semanticProgressVectorsText"),
    progressModelRow: byId("semanticProgressModelRow"),
    progressModel: byId("semanticProgressModel"),
    progressModelText: byId("semanticProgressModelText"),
    progressRuntimeRow: byId("semanticProgressRuntimeRow"),
    progressRuntime: byId("semanticProgressRuntime"),
    progressRuntimeText: byId("semanticProgressRuntimeText"),
    progressStage: byId("semanticProgressStage"),
    progressTotal: byId("semanticProgressTotal"),
    progressTotalText: byId("semanticProgressTotalText"),
    cancelDownload: byId("semanticCancelButton"),
    manage: byId("semanticManage"),
    disable: byId("semanticDisableButton"),
    clear: byId("semanticClearButton"),
  };
}

export function createSemanticController({ metaUrl = "", photoCount = 0, photosSha256 = "", trackEvent = () => {}, onChange = () => {} } = {}) {
  const dom = queryDom();
  const available = Boolean(metaUrl) && Boolean(dom.bar) && typeof Worker !== "undefined";
  const state = {
    status: "idle",
    mode: "",
    device: "",
    message: "",
    meta: null,
    sets: {},
    preferImage: false,
    webgpu: null,
    hasData: false,
  };
  let webgpuCheck = null;
  let currentRun = null;
  let worker = null;
  let workerReady = false;
  let abortController = null;
  let runToken = 0;
  let requestSequence = 0;
  const pendingEmbeddings = new Map();
  const embeddings = new Map();
  // runtime is a 0–1 fraction (the worker counts decoded bytes). stage becomes
  // "initializing" when the model files are in and ORT builds the GPU session.
  const progress = { vectors: new Map(), model: new Map(), runtime: 0, stage: "", includeModel: false };

  // Readiness follows the last mode that finished loading, so an upgrade from
  // image-only to full keeps image similarity usable while it downloads.
  function textReady() {
    return state.mode === "full" && workerReady && Boolean(state.sets.text || state.sets.image);
  }

  function imageReady() {
    return Boolean(state.mode) && Boolean(state.sets.image);
  }

  // Unknown until meta.json is read; assume yes so the entry stays visible.
  function hasImageVectors() {
    return !state.meta || Boolean(state.meta.sets.image);
  }

  function emit() {
    renderBar();
    renderDialog();
    onChange();
  }

  function setStatus(status, message = "") {
    state.status = status;
    state.message = message;
    emit();
  }

  function resolveMetaUrl() {
    return new URL(metaUrl, document.baseURI).toString();
  }

  async function loadMeta() {
    if (state.meta) {
      return state.meta;
    }
    const response = await fetch(resolveMetaUrl(), { cache: "no-cache" });
    if (!response.ok) {
      throw new Error(`無法讀取向量資訊：HTTP ${response.status}`);
    }
    const meta = parseVectorMeta(await response.json());
    if (meta.rowCount !== photoCount) {
      throw new Error("向量資料與照片索引的筆數不一致，請重新整理頁面");
    }
    if (photosSha256 && meta.photosSha256 && meta.photosSha256 !== photosSha256) {
      throw new Error("向量資料與照片索引版本不一致，請重新整理頁面");
    }
    state.meta = meta;
    renderSizes();
    return meta;
  }

  function vectorUrl(set) {
    return new URL(set.file, resolveMetaUrl()).toString();
  }

  function vectorCacheName(set) {
    return `${vectorCachePrefix}${set.sha256}`;
  }

  function checkWebGpu() {
    webgpuCheck ??= detectWebGpu().then((supported) => {
      state.webgpu = supported;
      return supported;
    });
    return webgpuCheck;
  }

  // caches.open() would create an empty cache, so look it up with has() first.
  async function cachedVector(set) {
    if (!hasCacheApi() || !(await caches.has(vectorCacheName(set)))) {
      return null;
    }
    const cache = await caches.open(vectorCacheName(set));
    return (await cache.match(vectorUrl(set))) ?? null;
  }

  async function modelCacheEntries(modelId) {
    if (!hasCacheApi() || !modelId || !(await caches.has(modelCacheName))) {
      return { cache: null, requests: [] };
    }
    const cache = await caches.open(modelCacheName);
    const requests = (await cache.keys()).filter((request) => isModelCacheUrl(request.url, modelId));
    return { cache, requests };
  }

  async function refreshHasData() {
    let hasData = false;
    if (hasCacheApi()) {
      try {
        const names = await caches.keys();
        hasData = names.some((name) => name.startsWith(vectorCachePrefix));
        if (!hasData && state.meta) {
          hasData = (await modelCacheEntries(state.meta.queryModel.id)).requests.length > 0;
        } else if (!hasData) {
          hasData = names.includes(modelCacheName);
        }
      } catch {
        hasData = false;
      }
    }
    if (hasData !== state.hasData) {
      state.hasData = hasData;
      renderDialog();
    }
  }

  // Removes what a failed or cancelled run stored: the vector caches it was
  // filling and, when it loaded the model, this model's transformers-cache
  // entries. Vector sets that were already active before the run are kept.
  async function discardRun(run) {
    if (!run || !hasCacheApi()) {
      return;
    }
    try {
      await Promise.all(
        run.kinds
          .map((kind) => run.meta.sets[kind])
          .filter(Boolean)
          .map((set) => caches.delete(vectorCacheName(set))),
      );
      if (run.includeModel) {
        const { cache, requests } = await modelCacheEntries(run.meta.queryModel.id);
        await Promise.all(requests.map((request) => cache.delete(request)));
        if (cache && (await cache.keys()).length === 0) {
          await caches.delete(modelCacheName);
        }
      }
    } catch {
      // Leftovers only waste space; "清除已下載資料" still removes them.
    }
    await refreshHasData();
  }

  async function modelFilesCached(meta) {
    if (!hasCacheApi()) {
      return false;
    }
    const cache = await caches.open(modelCacheName);
    const matches = await Promise.all(modelFileUrls(meta.queryModel).map((url) => cache.match(url)));
    return matches.every(Boolean);
  }

  async function allAssetsCached(meta, mode) {
    const kinds = setsForMode(mode, meta);
    if (kinds.length === 0) {
      return false;
    }
    for (const kind of kinds) {
      const set = meta.sets[kind];
      if (!set || !(await cachedVector(set))) {
        return false;
      }
    }
    return mode === "full" ? modelFilesCached(meta) : true;
  }

  async function readStream(response, expectedBytes, onProgress) {
    if (!response.body) {
      const buffer = new Uint8Array(await response.arrayBuffer());
      onProgress(buffer.byteLength);
      return buffer;
    }
    const output = new Uint8Array(expectedBytes);
    const reader = response.body.getReader();
    let offset = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) {
        break;
      }
      if (offset + value.byteLength > expectedBytes) {
        await reader.cancel();
        throw new Error("向量檔大小超出預期");
      }
      output.set(value, offset);
      offset += value.byteLength;
      onProgress(offset);
    }
    if (offset !== expectedBytes) {
      throw new Error("向量檔下載不完整");
    }
    return output;
  }

  async function loadVectorSet(meta, kind, { allowNetwork, signal, token }) {
    const set = meta.sets[kind];
    if (!set) {
      throw new Error(`這次部署沒有 ${kind === "text" ? "文字" : "圖片"} 向量`);
    }
    const onProgress = (loaded) => {
      progress.vectors.set(kind, { loaded, total: set.bytes });
      renderProgress();
    };
    let bytes;
    const cached = await cachedVector(set);
    if (cached) {
      bytes = new Uint8Array(await cached.arrayBuffer());
      onProgress(bytes.byteLength);
    } else {
      if (!allowNetwork) {
        throw new Error("需要重新下載");
      }
      const response = await fetch(vectorUrl(set), { signal, cache: "no-store" });
      if (!response.ok) {
        throw new Error(`無法下載向量：HTTP ${response.status}`);
      }
      bytes = await readStream(response, set.bytes, onProgress);
      const digest = await sha256Hex(bytes);
      if (digest && digest !== set.sha256) {
        throw new Error("向量檔內容與 sha256 不符");
      }
      if (token !== runToken) {
        throw new CancelledError();
      }
      if (hasCacheApi()) {
        try {
          const cache = await caches.open(vectorCacheName(set));
          await cache.put(
            vectorUrl(set),
            new Response(bytes, { headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength) } }),
          );
          if (token !== runToken) {
            await caches.delete(vectorCacheName(set));
            throw new CancelledError();
          }
        } catch (error) {
          if (isCancelled(error)) {
            throw error;
          }
          // Quota errors only cost a re-download next time.
        }
      }
    }
    return parseVectorSet(bytes, meta);
  }

  function rejectPendingEmbeddings(message) {
    for (const { reject } of pendingEmbeddings.values()) {
      reject(new Error(message));
    }
    pendingEmbeddings.clear();
  }

  function stopWorker() {
    worker?.terminate();
    worker = null;
    workerReady = false;
    rejectPendingEmbeddings("查詢模型已停止");
  }

  function startWorker(meta, token) {
    stopWorker();
    const workerStartedAt = performance.now();
    const instance = new Worker(new URL("./semantic-worker.js", import.meta.url), { type: "module" });
    worker = instance;
    return new Promise((resolve, reject) => {
      instance.addEventListener("message", (event) => {
        if (worker !== instance) {
          return;
        }
        const message = event.data ?? {};
        if (message.type === "progress") {
          progress.model.set(message.file, { loaded: message.loaded, total: message.total });
          renderProgress();
        } else if (message.type === "runtime-progress") {
          progress.runtime = message.total > 0 ? Math.min(message.loaded / message.total, 1) : 0;
          renderProgress();
        } else if (message.type === "log") {
          reportTiming("log", { level: message.level, text: message.text });
        } else if (message.type === "stage") {
          progress.stage = message.stage;
          renderProgress();
        } else if (message.type === "ready") {
          workerReady = true;
          progress.runtime = 1;
          progress.stage = "";
          renderProgress();
          reportTiming("load", { total: Math.round(performance.now() - workerStartedAt), ...message.timings });
          resolve(message.device);
        } else if (message.type === "embedding") {
          const pending = pendingEmbeddings.get(message.requestId);
          pendingEmbeddings.delete(message.requestId);
          reportTiming("embed", { ms: message.ms });
          pending?.resolve(message.vector);
        } else if (message.type === "error") {
          if (message.requestId !== undefined) {
            const pending = pendingEmbeddings.get(message.requestId);
            pendingEmbeddings.delete(message.requestId);
            pending?.reject(new Error(message.message));
          } else {
            reject(new Error(message.message || "查詢模型載入失敗"));
          }
        }
      });
      instance.addEventListener("error", (event) => {
        if (worker === instance) {
          reject(new Error(event.message || "查詢模型執行環境載入失敗"));
        }
      });
      if (token !== runToken) {
        reject(new CancelledError());
        return;
      }
      instance.postMessage({
        type: "load",
        modelId: meta.queryModel.id,
        revision: meta.queryModel.revision,
        dtype: (semanticDebug && debugParams.get("semantic-dtype")) || meta.queryModel.dtype || "q4",
        debug: semanticDebug,
        placement: semanticDebug && debugParams.has("semantic-placement"),
        matmul: semanticDebug ? (debugParams.get("semantic-matmul") ?? "") : "",
        dims: meta.dims,
        queryPrefix: meta.queryPrefix,
      });
    });
  }

  function resetProgress(mode, kinds) {
    progress.vectors = new Map(kinds.map((kind) => [kind, { loaded: 0, total: state.meta?.sets[kind]?.bytes ?? estimatedBytes.vectorSet }]));
    progress.model = new Map();
    progress.runtime = 0;
    progress.stage = "";
    progress.includeModel = mode === "full";
  }

  // Loads every asset a mode needs. With allowNetwork false (automatic restore
  // after reload) it only succeeds when vectors come from the Cache API.
  // `storedPreference` overrides what is saved; a WebGPU-less restore passes
  // "full" so the degraded image mode never overwrites the user's choice.
  async function activate(mode, { allowNetwork, readyMessage = "", storedPreference = "" }) {
    const token = ++runToken;
    abortController?.abort();
    abortController = new AbortController();
    const { signal } = abortController;
    const startedAt = performance.now();
    const previous = { mode: state.mode, sets: { ...state.sets } };
    let stage = "meta";
    let run = null;
    try {
      if (allowNetwork) {
        state.status = "downloading";
      } else {
        state.status = "loading";
      }
      state.message = "";
      emit();
      const meta = await loadMeta();
      if (setsForMode(mode, meta).length === 0) {
        throw new Error("這次部署沒有可用的照片向量");
      }
      if (mode === "full" && !(await checkWebGpu())) {
        throw new Error(noWebGpuMessage);
      }
      const kinds = setsForMode(mode, meta).filter((kind) => !state.sets[kind]);
      run = { meta, kinds, includeModel: mode === "full" && !workerReady, previousSets: previous.sets };
      currentRun = run;
      resetProgress(mode, kinds);
      renderProgress();
      stage = "assets";
      const vectorJobs = kinds.map((kind) =>
        loadVectorSet(meta, kind, { allowNetwork, signal, token }).then((set) => {
          if (token === runToken) {
            state.sets = { ...state.sets, [kind]: set };
          }
        }),
      );
      const needsModel = mode === "full" && !workerReady;
      const modelJob = needsModel ? startWorker(meta, token).then((device) => {
        state.device = device;
      }) : Promise.resolve();
      await Promise.all([...vectorJobs, modelJob]);
      if (token !== runToken) {
        throw new CancelledError();
      }
      state.mode = mode;
      writeSemanticPreference(storedPreference || mode);
      abortController = null;
      currentRun = null;
      setStatus("ready", readyMessage);
      refreshHasData();
      trackEvent("semantic_assets_ready", {
        download_ms: Math.round(performance.now() - startedAt),
        device: mode === "full" ? state.device : "none",
        semantic_mode: mode,
        from_cache: !allowNetwork,
      });
      cleanupStaleVectorCaches(meta);
      if (allowNetwork && dom.dialog?.open) {
        dom.dialog.close();
      }
    } catch (error) {
      if (token !== runToken) {
        return;
      }
      abortController?.abort();
      abortController = null;
      if (mode === "full" && !workerReady) {
        stopWorker();
      }
      state.mode = previous.mode;
      state.sets = previous.sets;
      currentRun = null;
      // A failed run leaves nothing half-enabled: drop what it stored and fall
      // back to the mode that was active before it (usually none).
      writeSemanticPreference(storedPreference || previous.mode);
      await discardRun(run);
      if (isCancelled(error)) {
        setStatus(previous.mode ? "ready" : "idle");
        return;
      }
      trackEvent("semantic_assets_failed", { semantic_mode: mode, failure_stage: stage, from_cache: !allowNetwork });
      const prefix = allowNetwork ? "下載失敗" : "載入失敗";
      setStatus(previous.mode ? "ready" : "error", `${prefix}：${error.message}`);
    }
  }

  function cancelDownload() {
    runToken += 1;
    abortController?.abort();
    abortController = null;
    if (!workerReady) {
      stopWorker();
    }
    const run = currentRun;
    currentRun = null;
    // Sets loaded by the cancelled run are discarded with their caches.
    if (run) {
      state.sets = run.previousSets;
    }
    writeSemanticPreference(state.mode);
    setStatus(state.mode ? "ready" : "idle", "已取消下載，這次下載的部分資料已清除。");
    discardRun(run);
  }

  async function cleanupStaleVectorCaches(meta) {
    if (!hasCacheApi()) {
      return;
    }
    try {
      const keep = new Set(Object.values(meta.sets).map((set) => vectorCacheName(set)));
      const names = await caches.keys();
      await Promise.all(names.filter((name) => name.startsWith(vectorCachePrefix) && !keep.has(name)).map((name) => caches.delete(name)));
    } catch {
      // Stale caches only waste space.
    }
  }

  function disable() {
    runToken += 1;
    abortController?.abort();
    abortController = null;
    stopWorker();
    state.sets = {};
    state.mode = "";
    state.device = "";
    embeddings.clear();
    writeSemanticPreference("");
    setStatus("idle", "語意搜尋已關閉，下載的資料仍保留在瀏覽器中。");
  }

  async function clearDownloadedData() {
    disable();
    if (hasCacheApi()) {
      try {
        const names = await caches.keys();
        await Promise.all(names.filter((name) => name.startsWith(vectorCachePrefix) || name === modelCacheName).map((name) => caches.delete(name)));
      } catch {
        setStatus("idle", "無法清除部分下載資料，請改用瀏覽器的網站資料設定清除。");
        return;
      }
    }
    setStatus("idle", "已清除語意搜尋下載的資料。");
    await refreshHasData();
  }

  function embed(text) {
    const key = String(text ?? "").trim();
    if (embeddings.has(key)) {
      return Promise.resolve(embeddings.get(key));
    }
    if (!worker || !workerReady) {
      return Promise.reject(new Error("查詢模型尚未就緒"));
    }
    requestSequence += 1;
    const requestId = requestSequence;
    return new Promise((resolve, reject) => {
      pendingEmbeddings.set(requestId, {
        resolve: (vector) => {
          embeddings.set(key, vector);
          if (embeddings.size > embeddingCacheSize) {
            embeddings.delete(embeddings.keys().next().value);
          }
          resolve(vector);
        },
        reject,
      });
      worker.postMessage({ type: "embed", requestId, text: key });
    });
  }

  function cachedEmbedding(text) {
    return embeddings.get(String(text ?? "").trim()) ?? null;
  }

  function progressTotals() {
    let loaded = 0;
    let total = 0;
    let vectorLoaded = 0;
    let vectorTotal = 0;
    for (const item of progress.vectors.values()) {
      vectorLoaded += item.loaded;
      vectorTotal += item.total;
    }
    loaded += vectorLoaded;
    total += vectorTotal;
    let modelLoaded = 0;
    let modelTotal = 0;
    for (const item of progress.model.values()) {
      modelLoaded += item.loaded;
      modelTotal += item.total;
    }
    if (progress.includeModel) {
      modelTotal = workerReady ? modelLoaded || modelTotal : Math.max(modelTotal, estimatedBytes.model);
      if (workerReady) {
        modelLoaded = modelTotal;
      }
      loaded += modelLoaded + progress.runtime * estimatedBytes.runtime;
      total += modelTotal + estimatedBytes.runtime;
    }
    return { loaded, total, vectorLoaded, vectorTotal, modelLoaded, modelTotal };
  }

  function setProgress(element, text, loaded, total) {
    if (!element) {
      return;
    }
    element.max = Math.max(total, 1);
    element.value = Math.min(loaded, total);
    if (text) {
      text.textContent = `${formatMegabytes(loaded)} / ${formatMegabytes(total)}`;
    }
  }

  function renderProgress() {
    if (!dom.progress) {
      return;
    }
    const totals = progressTotals();
    setProgress(dom.progressVectors, dom.progressVectorsText, totals.vectorLoaded, totals.vectorTotal);
    dom.progressModelRow.hidden = !progress.includeModel;
    dom.progressRuntimeRow.hidden = !progress.includeModel;
    if (progress.includeModel) {
      setProgress(dom.progressModel, dom.progressModelText, totals.modelLoaded, totals.modelTotal);
      setProgress(dom.progressRuntime, null, progress.runtime, 1);
      dom.progressRuntimeText.textContent = progress.runtime >= 1 ? "完成" : `${Math.floor(progress.runtime * 100)}%`;
      // After the downloads finish, building the WebGPU session and compiling
      // shaders still takes a few seconds with no byte progress to show.
      const initializing = !workerReady && progress.stage === "initializing";
      dom.progressStage.hidden = !initializing;
      if (initializing) {
        dom.progressStage.textContent = progress.runtime >= 1 ? "下載完成，正在初始化 GPU…" : "模型下載完成，等待執行環境…";
      }
    }
    setProgress(dom.progressTotal, dom.progressTotalText, totals.loaded, totals.total);
    if (state.status === "downloading" || state.status === "loading") {
      renderBar();
    }
  }

  function renderSizes() {
    const sets = state.meta?.sets;
    const vectorBytes = sets
      ? Object.values(sets).reduce((total, set) => total + set.bytes, 0)
      : estimatedBytes.vectorSet * 2;
    const imageBytes = sets?.image?.bytes ?? estimatedBytes.vectorSet;
    // Without WebGPU only image similarity is possible, so list just its data.
    const imageOnly = state.webgpu === false;
    dom.sizeVectorsLabel.textContent = imageOnly ? "圖片向量" : "照片向量（文字與圖片）";
    dom.sizeModelRow.hidden = imageOnly;
    dom.sizeRuntimeRow.hidden = imageOnly;
    dom.sizeTotalRow.hidden = imageOnly;
    dom.sizeVectors.textContent = `約 ${formatMegabytes(imageOnly ? imageBytes : vectorBytes)}`;
    dom.sizeModel.textContent = `約 ${formatMegabytes(estimatedBytes.model)}`;
    dom.sizeRuntime.textContent = `約 ${formatMegabytes(estimatedBytes.runtime)}`;
    dom.sizeTotal.textContent = `約 ${formatMegabytes(vectorBytes + estimatedBytes.model + estimatedBytes.runtime)}`;
    dom.enableImage.textContent = `只下載以圖找圖（約 ${formatMegabytes(imageBytes)}）`;
    renderModelInfo();
  }

  // The dialog names the exact models from the deployed meta.json, so the
  // disclosure follows the data when the model or its revision changes.
  function renderModelInfo() {
    const meta = state.meta;
    if (!meta) {
      return;
    }
    if (meta.sourceModel) {
      dom.sourceModel.textContent = meta.sourceModel;
    }
    const { id, revision, dtype } = meta.queryModel;
    if (id) {
      dom.queryModelLink.textContent = id;
      dom.queryModelLink.href = `${modelHost}/${id}${revision ? `/tree/${encodeURIComponent(revision)}` : ""}`;
    }
    dom.queryModelDetail.textContent = [dtype, revision ? `版本 ${revision.slice(0, 7)}` : ""].filter(Boolean).join("，");
  }

  function statusText() {
    const totals = progressTotals();
    const percent = totals.total > 0 ? Math.floor((totals.loaded / totals.total) * 100) : 0;
    switch (state.status) {
      case "checking":
        return "正在確認瀏覽器中的語意搜尋資料…";
      case "loading":
        return "正在從瀏覽器快取載入語意搜尋…";
      case "downloading":
        if (progress.stage === "initializing" && !workerReady && totals.loaded >= totals.total) {
          return "語意搜尋資料下載完成，正在初始化 GPU…";
        }
        return `正在下載語意搜尋資料 ${percent}%`;
      case "ready":
        if (state.mode === "full") {
          return "語意搜尋已開啟（WebGPU）。輸入描述即可查詢，查詢內容只在本機計算。";
        }
        return `${state.message ? `${state.message} ` : ""}以圖找圖已開啟：在照片預覽按「找相似照片」。${state.webgpu === false ? "" : "文字語意搜尋尚未下載。"}`;
      case "needs-download":
        return "語意搜尋的資料已不在瀏覽器中，需要重新下載。";
      case "error":
        return state.message || "語意搜尋載入失敗。";
      default:
        return "";
    }
  }

  function renderBar() {
    if (!dom.bar) {
      return;
    }
    dom.bar.hidden = !available;
    if (!available) {
      return;
    }
    const text = statusText();
    dom.status.textContent = text;
    dom.status.hidden = !text;
    dom.status.dataset.status = state.status;
    dom.button.classList.toggle("is-active", state.status === "ready");
    dom.button.setAttribute("aria-expanded", dom.dialog.open ? "true" : "false");
  }

  function renderDialog() {
    if (!dom.dialog) {
      return;
    }
    const busy = state.status === "downloading" || state.status === "loading";
    dom.choice.hidden = busy;
    dom.progress.hidden = !busy;
    const fullReady = state.status === "ready" && state.mode === "full";
    const imageOnly = state.status === "ready" && state.mode === "image";
    const noWebGpu = state.webgpu === false;
    dom.enableFull.hidden = fullReady;
    dom.enableFull.disabled = noWebGpu;
    dom.enableImage.hidden = fullReady || imageOnly || !hasImageVectors();
    dom.enableFull.textContent = imageOnly ? "下載文字語意搜尋" : "下載並開啟語意搜尋";
    dom.enableFull.classList.toggle("is-primary", !noWebGpu && (!state.preferImage || imageOnly));
    dom.enableImage.classList.toggle("is-primary", noWebGpu || (state.preferImage && !imageOnly));
    dom.webgpuNote.hidden = !noWebGpu || fullReady;
    dom.webgpuNote.textContent = noWebGpuMessage;
    dom.disable.hidden = state.status !== "ready";
    dom.clear.hidden = !state.hasData;
    dom.manage.hidden = busy || (dom.disable.hidden && dom.clear.hidden);
    dom.message.textContent = state.message || (state.status === "needs-download" ? statusText() : "");
    dom.message.hidden = !dom.message.textContent;
  }

  function openDialog({ preferImage = false } = {}) {
    if (!available) {
      return;
    }
    state.preferImage = preferImage;
    if (state.status === "error" || state.status === "idle") {
      state.message = "";
    }
    renderSizes();
    renderDialog();
    if (!dom.dialog.open) {
      dom.dialog.showModal();
    }
    renderBar();
    const focusTarget = preferImage && !dom.enableImage.hidden ? dom.enableImage : null;
    (focusTarget ?? dom.dismiss).focus({ preventScroll: true });
    loadMeta().then(() => {
      renderSizes();
      renderDialog();
      onChange();
      refreshHasData();
    }, () => {});
    checkWebGpu().then(() => {
      renderSizes();
      renderDialog();
    });
    refreshHasData();
  }

  async function enable(mode) {
    if (mode === "full" && !(await checkWebGpu())) {
      renderDialog();
      return;
    }
    trackEvent("enable_semantic_search", { semantic_mode: mode });
    activate(mode, { allowNetwork: true });
  }

  // Automatic restore after reload never downloads. A stored "full" mode on a
  // browser without WebGPU degrades to image similarity when its vectors are
  // cached, otherwise the preference is dropped with an explanation.
  async function restore() {
    const preference = readSemanticPreference();
    if (!available || !preference) {
      return;
    }
    state.status = "checking";
    emit();
    let mode = preference;
    let readyMessage = "";
    let storedPreference = preference;
    try {
      const meta = await loadMeta();
      const webgpu = preference === "full" ? await checkWebGpu() : false;
      const imageCached = Boolean(meta.sets.image) && Boolean(await cachedVector(meta.sets.image));
      const resolved = resolveSemanticMode({ preference, webgpu, imageCached });
      storedPreference = resolved.storedPreference;
      if (!resolved.mode) {
        setStatus("error", noWebGpuMessage);
        return;
      }
      mode = resolved.mode;
      if (resolved.reason === "no-webgpu") {
        readyMessage = "這個瀏覽器不支援 WebGPU，已改為只用以圖找圖。";
      }
      if (!(await allAssetsCached(meta, mode))) {
        setStatus("needs-download");
        return;
      }
    } catch (error) {
      setStatus("needs-download", error.message);
      return;
    }
    await activate(mode, { allowNetwork: false, readyMessage, storedPreference });
  }

  if (available) {
    dom.button.addEventListener("click", () => openDialog());
    dom.enableFull.addEventListener("click", () => enable("full"));
    dom.enableImage.addEventListener("click", () => enable("image"));
    dom.dismiss.addEventListener("click", () => dom.dialog.close());
    dom.cancelDownload.addEventListener("click", cancelDownload);
    dom.disable.addEventListener("click", disable);
    dom.clear.addEventListener("click", clearDownloadedData);
    dom.dialog.addEventListener("close", renderBar);
    dom.dialog.addEventListener("click", (event) => {
      if (event.target !== dom.dialog) {
        return;
      }
      const rect = dom.dialog.getBoundingClientRect();
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
        dom.dialog.close();
      }
    });
  }
  renderBar();

  return {
    get available() {
      return available;
    },
    get status() {
      return state.status;
    },
    get sets() {
      return state.sets;
    },
    get dialog() {
      return dom.dialog;
    },
    get device() {
      return state.device;
    },
    textReady,
    imageReady,
    hasImageVectors,
    embed,
    cachedEmbedding,
    openDialog,
    restore,
    disable,
    clearDownloadedData,
  };
}
