// Web Worker for opt-in semantic search. It loads only the text half of the
// pinned EmbeddingGemma 2 ONNX model and turns query text into a normalized
// vector. Queries never leave the browser; only model files are downloaded.
const transformersUrl = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.2/dist/transformers.min.js";
const modelHost = "https://huggingface.co";

let model = null;
let tokenizer = null;
let settings = null;
let loadPromise = null;

function post(message, transfer = []) {
  self.postMessage(message, transfer);
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

async function fetchTextConfig({ modelId, revision }) {
  const url = `${modelHost}/${modelId}/resolve/${encodeURIComponent(revision)}/config.json`;
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`無法讀取模型設定：HTTP ${response.status}`);
  }
  const config = await response.json();
  // Removing the vision and audio sub-configs keeps transformers.js from
  // downloading those encoders; the query side only needs text.
  delete config.vision_config;
  delete config.audio_config;
  return config;
}

function progressCallback(info) {
  if (info?.status === "progress" && info.file) {
    post({ type: "progress", file: info.file, loaded: Number(info.loaded) || 0, total: Number(info.total) || 0 });
  }
}

async function loadModel(message) {
  settings = {
    modelId: message.modelId,
    revision: message.revision,
    dtype: message.dtype || "q4",
    dims: Number(message.dims) || 512,
    queryPrefix: String(message.queryPrefix ?? ""),
  };
  if (typeof navigator === "undefined" || !navigator.gpu) {
    throw new Error("這個瀏覽器不支援 WebGPU，無法執行文字語意搜尋");
  }
  const { AutoModel, AutoTokenizer, env } = await import(transformersUrl);
  // Pin every Hub request, including transformers.js internal probes that do
  // not forward `revision`, to the fixed model revision.
  env.remotePathTemplate = `{model}/resolve/${encodeURIComponent(settings.revision)}/`;
  const config = await fetchTextConfig(settings);
  tokenizer = await AutoTokenizer.from_pretrained(settings.modelId, {
    revision: settings.revision,
    progress_callback: progressCallback,
  });

  // WebGPU only: every quantized export uses GatherBlockQuantized, which the
  // ORT-web WASM backend does not implement.
  model = await AutoModel.from_pretrained(settings.modelId, {
    config,
    dtype: settings.dtype,
    revision: settings.revision,
    device: "webgpu",
    progress_callback: progressCallback,
  });
  // A first embedding confirms the device can actually run the graph.
  await embedText("warmup");
  return "webgpu";
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
      post({ type: "ready", device });
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
      const vector = await embedText(String(message.text ?? ""));
      post({ type: "embedding", requestId: message.requestId, vector }, [vector.buffer]);
    } catch (error) {
      post({ type: "error", requestId: message.requestId, message: errorMessage(error) });
    }
  }
});
