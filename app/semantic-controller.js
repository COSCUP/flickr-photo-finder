// Opt-in semantic search lifecycle: consent dialog, downloads with progress,
// Cache API reuse, the query worker, and the per-device enabled preference.
// Ranking stays in semantic-search.js; this module owns network and storage.
import { assembleVectorRows, parseVectorMeta, parseVectorSet, readEncodedBody } from "./semantic-search.js";

export const semanticStorageKey = "coscup-photo-finder-semantic";
// Must not start with "coscup-photo-finder-": the service worker deletes those
// caches on every deployment, while vector caches should survive unchanged.
export const vectorCachePrefix = "photo-finder-semantic-vectors-";
export const modelCacheName = "transformers-cache";
export const modelCacheFiles = ["onnx/model_q4.onnx", "onnx/model_q4.onnx_data", "tokenizer.json", "tokenizer_config.json"];
const modelHost = "https://huggingface.co";
// Transfer sizes. The runtime (transformers.js plus the ORT WASM binary) comes
// compressed from jsDelivr: about 5.7 MB on the wire, 27 MB once decoded. The
// model weights come uncompressed from huggingface.co; the tokenizer comes
// gzipped from this site when meta.json lists it (5.4 MB instead of 32 MB).
const estimatedBytes = { modelWeights: 175_000_000, tokenizer: 32_200_000, runtime: 5_700_000, vectorSet: 13_300_000 };
const embeddingCacheSize = 32;

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
// `semantic-backend=wasm|webgpu` forces the query-model backend; all need
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

// One decimal below 10 MB so small segment updates do not read as "1 MB".
export function formatMegabytes(bytes) {
  const megabytes = bytes / 1_000_000;
  return megabytes < 10 ? `${Math.max(0.1, Math.round(megabytes * 10) / 10)} MB` : `${Math.round(megabytes)} MB`;
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
    choice: byId("semanticChoice"),
    sizeVectors: byId("semanticSizeVectors"),
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

// `photoIds` is the photos-index order; vectors are matched to it by photo_id.
export function createSemanticController({ metaUrl = "", photoIds = [], trackEvent = () => {}, onChange = () => {} } = {}) {
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
    hasData: false,
  };
  let currentRun = null;
  let worker = null;
  let workerReady = false;
  let abortController = null;
  let runToken = 0;
  let requestSequence = 0;
  const pendingEmbeddings = new Map();
  const embeddings = new Map();
  // runtime is a 0–1 fraction (the worker counts decoded bytes). stage becomes
  // "initializing" when the model files are in and ORT builds the session.
  const progress = { vectors: new Map(), model: new Map(), runtime: 0, stage: "", includeModel: false, modelCached: false };
  // What this browser already has for the deployed meta.json: bytes still
  // missing per vector kind and whether the query model files are cached.
  // The dialog lists and the progress bars count only what is missing.
  let cacheInfo = null;

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
    state.meta = meta;
    renderSizes();
    return meta;
  }

  function vectorUrl(path) {
    return new URL(path, resolveMetaUrl()).toString();
  }

  // Query model download: weights from huggingface.co plus the tokenizer,
  // gzipped from this site when the deployment ships it.
  function modelEstimate() {
    return estimatedBytes.modelWeights + (state.meta?.queryModel.tokenizer?.transferBytes ?? estimatedBytes.tokenizer);
  }

  // One cache per immutable segment, named by the sha256 of its rows, holding
  // the decoded rows and its photo_id list. A new deployment reuses every
  // segment the browser already has and downloads only the new ones.
  function segmentCacheName(segment) {
    return `${vectorCachePrefix}${segment.sha256}`;
  }

  // caches.open() would create an empty cache, so look it up with has() first.
  async function cachedSegment(segment) {
    if (!hasCacheApi() || !(await caches.has(segmentCacheName(segment)))) {
      return null;
    }
    const cache = await caches.open(segmentCacheName(segment));
    const [rows, ids] = await Promise.all([cache.match(vectorUrl(segment.file)), cache.match(vectorUrl(segment.ids))]);
    if (!rows || !ids) {
      return null;
    }
    const bytes = new Uint8Array(await rows.arrayBuffer());
    const photoIds = await ids.json();
    return bytes.byteLength === segment.bytes && Array.isArray(photoIds) && photoIds.length === segment.count ? { bytes, photoIds } : null;
  }

  function allSegments(meta) {
    return Object.values(meta.sets).flatMap((set) => set.segments);
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
      // Only segments this run stored; ones cached before it stay usable.
      await Promise.all([...run.storedSegments].map((name) => caches.delete(name)));
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

  async function segmentCacheState(meta, mode) {
    let missingBytes = 0;
    let cachedSegments = 0;
    for (const kind of setsForMode(mode, meta)) {
      for (const segment of meta.sets[kind]?.segments ?? []) {
        if (await cachedSegment(segment)) {
          cachedSegments += 1;
        } else {
          missingBytes += segment.transferBytes + segment.idsBytes;
        }
      }
    }
    return { missingBytes, cachedSegments };
  }

  async function allAssetsCached(meta, mode) {
    const kinds = setsForMode(mode, meta);
    if (kinds.length === 0) {
      return false;
    }
    for (const kind of kinds) {
      const set = meta.sets[kind];
      for (const segment of set?.segments ?? [null]) {
        if (!segment || !(await cachedSegment(segment))) {
          return false;
        }
      }
    }
    return mode === "full" ? modelFilesCached(meta) : true;
  }

  async function downloadSegment(segment, { signal, token, run, onProgress }) {
    const [rows, ids] = await Promise.all([
      fetch(vectorUrl(segment.file), { signal, cache: "no-store" }),
      fetch(vectorUrl(segment.ids), { signal, cache: "no-store" }),
    ]);
    if (!rows.ok || !ids.ok) {
      throw new Error(`無法下載向量：HTTP ${rows.ok ? ids.status : rows.status}`);
    }
    const [bytes, photoIds] = await Promise.all([readEncodedBody(rows, { expectedBytes: segment.bytes, onProgress }), ids.json()]);
    const digest = await sha256Hex(bytes);
    if (digest && digest !== segment.sha256) {
      throw new Error("向量檔內容與 sha256 不符");
    }
    if (!Array.isArray(photoIds) || photoIds.length !== segment.count) {
      throw new Error("向量分段的 photo_id 清單不正確");
    }
    if (token !== runToken) {
      throw new CancelledError();
    }
    if (hasCacheApi()) {
      const name = segmentCacheName(segment);
      try {
        // Start from an empty cache so entries under other URLs never linger.
        await caches.delete(name);
        const cache = await caches.open(name);
        run?.storedSegments.add(name);
        await cache.put(vectorUrl(segment.file), new Response(bytes, { headers: { "content-type": "application/octet-stream", "content-length": String(bytes.byteLength) } }));
        await cache.put(vectorUrl(segment.ids), new Response(JSON.stringify(photoIds), { headers: { "content-type": "application/json" } }));
        if (token !== runToken) {
          throw new CancelledError();
        }
      } catch (error) {
        if (isCancelled(error)) {
          throw error;
        }
        // Quota errors only cost a re-download next time.
      }
    }
    return { bytes, photoIds };
  }

  // Merges a kind's segments in order; cached segments are reused and only the
  // missing ones are downloaded. Progress counts transfer bytes.
  async function loadVectorSet(meta, kind, { allowNetwork, signal, token, run }) {
    const set = meta.sets[kind];
    if (!set) {
      throw new Error(`這次部署沒有 ${kind === "text" ? "文字" : "圖片"} 向量`);
    }
    const segments = await Promise.all(set.segments.map((segment) => cachedSegment(segment)));
    const total = set.segments.reduce((sum, segment, index) => sum + (segments[index] ? 0 : segment.transferBytes + segment.idsBytes), 0);
    let done = 0;
    const report = (current) => {
      progress.vectors.set(kind, { loaded: Math.min(done + current, total), total });
      renderProgress();
    };
    report(0);
    for (const [index, segment] of set.segments.entries()) {
      if (segments[index]) {
        continue;
      }
      if (!allowNetwork) {
        throw new Error("需要重新下載");
      }
      segments[index] = await downloadSegment(segment, { signal, token, run, onProgress: report });
      done += segment.transferBytes + segment.idsBytes;
      report(0);
    }
    const { data } = assembleVectorRows({ segments, photoIds, rowBytes: meta.rowBytes });
    return parseVectorSet(data, { dims: meta.dims, rowBytes: meta.rowBytes, rowCount: photoIds.length });
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
        tokenizer: meta.queryModel.tokenizer ? { ...meta.queryModel.tokenizer, url: vectorUrl(meta.queryModel.tokenizer.file) } : null,
        debug: semanticDebug,
        placement: semanticDebug && debugParams.has("semantic-placement"),
        backend: semanticDebug ? (debugParams.get("semantic-backend") ?? "") : "",
        threads: semanticDebug ? Number(debugParams.get("semantic-threads")) || 0 : 0,
        dims: meta.dims,
        queryPrefix: meta.queryPrefix,
      });
    });
  }

  function resetProgress(mode, kinds, { modelCached = false } = {}) {
    progress.vectors = new Map(kinds.map((kind) => [kind, { loaded: 0, total: 0 }]));
    progress.model = new Map();
    progress.runtime = 0;
    progress.stage = "";
    progress.includeModel = mode === "full";
    progress.modelCached = modelCached;
  }

  async function refreshCacheInfo(meta) {
    const missing = {};
    for (const kind of Object.keys(meta.sets)) {
      missing[kind] = 0;
      for (const segment of meta.sets[kind].segments) {
        if (!(await cachedSegment(segment))) {
          missing[kind] += segment.transferBytes + segment.idsBytes;
        }
      }
    }
    cacheInfo = { missing, modelCached: await modelFilesCached(meta).catch(() => false) };
    return cacheInfo;
  }

  // Loads every asset a mode needs. With allowNetwork false (automatic restore
  // after reload) it only succeeds when vectors come from the Cache API.
  async function activate(mode, { allowNetwork }) {
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
      const kinds = setsForMode(mode, meta).filter((kind) => !state.sets[kind]);
      run = { meta, kinds, includeModel: mode === "full" && !workerReady, previousSets: previous.sets, storedSegments: new Set() };
      currentRun = run;
      resetProgress(mode, kinds, { modelCached: mode === "full" && (await modelFilesCached(meta).catch(() => false)) });
      renderProgress();
      stage = "assets";
      const vectorJobs = kinds.map((kind) =>
        loadVectorSet(meta, kind, { allowNetwork, signal, token, run }).then((set) => {
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
      writeSemanticPreference(mode);
      abortController = null;
      currentRun = null;
      setStatus("ready");
      refreshHasData();
      trackEvent("semantic_assets_ready", {
        download_ms: Math.round(performance.now() - startedAt),
        device: mode === "full" ? state.device : "none",
        semantic_mode: mode,
        from_cache: !allowNetwork,
      });
      cacheInfo = null;
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
      writeSemanticPreference(previous.mode);
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
      const keep = new Set(allSegments(meta).map((segment) => segmentCacheName(segment)));
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
    cacheInfo = null;
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
    if (progress.includeModel && !progress.modelCached) {
      modelTotal = workerReady ? modelLoaded || modelTotal : Math.max(modelTotal, modelEstimate());
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
    if (totals.vectorTotal === 0) {
      setProgress(dom.progressVectors, null, 1, 1);
      dom.progressVectorsText.textContent = "已下載";
    }
    dom.progressModelRow.hidden = !progress.includeModel;
    dom.progressRuntimeRow.hidden = !progress.includeModel;
    if (progress.includeModel && progress.modelCached) {
      setProgress(dom.progressModel, null, 1, 1);
      dom.progressModelText.textContent = "已下載";
      setProgress(dom.progressRuntime, null, 1, 1);
      dom.progressRuntimeText.textContent = "已下載";
    } else if (progress.includeModel) {
      setProgress(dom.progressModel, dom.progressModelText, totals.modelLoaded, totals.modelTotal);
      setProgress(dom.progressRuntime, null, progress.runtime, 1);
      dom.progressRuntimeText.textContent = progress.runtime >= 1 ? "完成" : `${Math.floor(progress.runtime * 100)}%`;
      // After the downloads finish, building the inference session (and
      // compiling shaders on WebGPU) still takes a few seconds with no byte
      // progress to show.
      const initializing = !workerReady && progress.stage === "initializing";
      dom.progressStage.hidden = !initializing;
      if (initializing) {
        dom.progressStage.textContent = progress.runtime >= 1 ? "下載完成，正在初始化查詢模型…" : "模型下載完成，等待執行環境…";
      }
    }
    setProgress(dom.progressTotal, dom.progressTotalText, totals.loaded, totals.total);
    if (totals.total === 0) {
      setProgress(dom.progressTotal, null, 1, 1);
      dom.progressTotalText.textContent = "已下載";
    }
    if (state.status === "downloading" || state.status === "loading") {
      renderBar();
    }
  }

  // Sizes list only what this browser still has to download.
  function renderSizes() {
    const sets = state.meta?.sets;
    const missingOf = (kind) => cacheInfo?.missing[kind] ?? sets?.[kind]?.transferBytes ?? estimatedBytes.vectorSet;
    const vectorBytes = sets ? Object.keys(sets).reduce((total, kind) => total + missingOf(kind), 0) : estimatedBytes.vectorSet * 2;
    const imageBytes = missingOf("image");
    const modelBytes = cacheInfo?.modelCached ? 0 : modelEstimate();
    const runtimeBytes = cacheInfo?.modelCached ? 0 : estimatedBytes.runtime;
    const size = (bytes) => (bytes > 0 ? `約 ${formatMegabytes(bytes)}` : "已下載");
    dom.sizeVectors.textContent = size(vectorBytes);
    dom.sizeModel.textContent = size(modelBytes);
    dom.sizeRuntime.textContent = size(runtimeBytes);
    dom.sizeTotal.textContent = size(vectorBytes + modelBytes + runtimeBytes);
    dom.enableImage.textContent = imageBytes > 0 ? `只下載以圖找圖（約 ${formatMegabytes(imageBytes)}）` : "只開啟以圖找圖";
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
          return "語意搜尋資料下載完成，正在初始化查詢模型…";
        }
        return `正在下載語意搜尋資料 ${percent}%`;
      case "ready":
        if (state.mode === "full") {
          return `語意搜尋已開啟（${state.device === "wasm" ? "CPU" : "WebGPU"}）。輸入描述即可查詢，查詢內容只在本機計算。`;
        }
        return `${state.message ? `${state.message} ` : ""}以圖找圖已開啟：在照片預覽按「找相似照片」。文字語意搜尋尚未下載。`;
      case "needs-download":
        return state.message || "語意搜尋的資料已不在瀏覽器中，需要重新下載。";
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
    dom.enableFull.hidden = fullReady;
    dom.enableImage.hidden = fullReady || imageOnly || !hasImageVectors();
    dom.enableFull.textContent = imageOnly ? "下載文字語意搜尋" : "下載並開啟語意搜尋";
    dom.enableFull.classList.toggle("is-primary", !state.preferImage || imageOnly);
    dom.enableImage.classList.toggle("is-primary", state.preferImage && !imageOnly);
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
    loadMeta().then(async (meta) => {
      await refreshCacheInfo(meta).catch(() => {});
      renderSizes();
      renderDialog();
      onChange();
      refreshHasData();
    }, () => {});
    refreshHasData();
  }

  function enable(mode) {
    trackEvent("enable_semantic_search", { semantic_mode: mode });
    activate(mode, { allowNetwork: true });
  }

  // Automatic restore after reload never downloads.
  async function restore() {
    const preference = readSemanticPreference();
    if (!available || !preference) {
      return;
    }
    state.status = "checking";
    emit();
    try {
      const meta = await loadMeta();
      if (!(await allAssetsCached(meta, preference))) {
        // A deployment that only added vector segments needs a small update,
        // not a full download; say how much so the visitor can decide.
        const { missingBytes, cachedSegments } = await segmentCacheState(meta, preference);
        const modelReady = preference !== "full" || (await modelFilesCached(meta));
        const update = modelReady && cachedSegments > 0;
        setStatus("needs-download", update ? `語意搜尋資料有更新，需要下載約 ${formatMegabytes(missingBytes)}。` : "");
        return;
      }
    } catch (error) {
      setStatus("needs-download", error.message);
      return;
    }
    await activate(preference, { allowNetwork: false });
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
