// Local embedding server adapters for pnpm finder:embed.
// Both engines expose an OpenAI-style POST /v1/embeddings; they differ in how
// multimodal input is sent. Responses are raw vectors (768d for EmbeddingGemma 2).

export const defaultEmbeddingEngine = "mlx-serve";
export const embeddingEngines = new Set(["mlx-serve", "omlx"]);
export const embeddingEngineDefaults = {
  "mlx-serve": { url: "http://127.0.0.1:11234", model: "mlx-community/embeddinggemma-2-8bit" },
  omlx: { url: "http://127.0.0.1:8000", model: "embeddinggemma-2-8bit" },
};
export const embeddingApiKeyEnv = "FINDER_EMBED_API_KEY";
// mlx-serve embeds one image per request and processes requests serially, so
// more than two in flight does not help.
export const mlxServeImageConcurrency = 2;
const defaultTimeoutMs = 300000;

export function embeddingsEndpoint(baseUrl) {
  const url = new URL(baseUrl);
  if (/\/embeddings\/?$/.test(url.pathname)) {
    return url.toString();
  }
  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = /\/v1$/.test(path) ? `${path}/embeddings` : `${path}/v1/embeddings`;
  return url.toString();
}

export function jpegDataUrl(buffer) {
  return `data:image/jpeg;base64,${Buffer.from(buffer).toString("base64")}`;
}

// omlx: items carry either text or an image data URI (omlx/api/embedding_models.py).
export function omlxTextRequestBody(model, texts) {
  return { model, items: texts.map((text) => ({ text })), encoding_format: "float" };
}

export function omlxImageRequestBody(model, imageDataUrls) {
  return { model, items: imageDataUrls.map((image) => ({ image })), encoding_format: "float" };
}

export function mlxServeTextRequestBody(model, texts) {
  return { model, input: texts };
}

// mlx-serve returns a single vector per messages request, so one image per call.
export function mlxServeImageRequestBody(model, imageDataUrl) {
  return {
    model,
    messages: [{ role: "user", content: [{ type: "image_url", image_url: { url: imageDataUrl } }] }],
  };
}

export function parseEmbeddingResponse(payload, expectedCount) {
  const data = Array.isArray(payload?.data) ? [...payload.data] : null;
  if (!data) {
    throw new Error("Embedding response does not contain a data array");
  }
  if (data.length !== expectedCount) {
    throw new Error(`Embedding response has ${data.length} vectors; expected ${expectedCount}`);
  }
  data.sort((left, right) => (left.index ?? 0) - (right.index ?? 0));
  return data.map((item) => {
    if (!Array.isArray(item.embedding)) {
      throw new Error("Embedding response item does not contain a float embedding array");
    }
    return Float32Array.from(item.embedding);
  });
}

async function mapWithConcurrency(items, concurrency, worker) {
  const results = new Array(items.length);
  let next = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (next < items.length) {
      const index = next;
      next += 1;
      results[index] = await worker(items[index]);
    }
  });
  await Promise.all(runners);
  return results;
}

export function createEmbeddingClient({
  engine = defaultEmbeddingEngine,
  url = "",
  model = "",
  apiKey = "",
  fetchImpl = fetch,
  timeoutMs = defaultTimeoutMs,
} = {}) {
  if (!embeddingEngines.has(engine)) {
    throw new Error(`--engine must be one of: ${[...embeddingEngines].join(", ")}`);
  }
  const defaults = embeddingEngineDefaults[engine];
  const endpoint = embeddingsEndpoint(url || defaults.url);
  const resolvedModel = model || defaults.model;

  async function post(body, expectedCount) {
    const headers = { "content-type": "application/json" };
    if (apiKey) {
      headers.authorization = `Bearer ${apiKey}`;
    }
    const response = await fetchImpl(endpoint, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) {
      const detail = (await response.text().catch(() => "")).slice(0, 300);
      throw new Error(`${engine} embeddings HTTP ${response.status}${detail ? `: ${detail}` : ""}`);
    }
    return parseEmbeddingResponse(await response.json(), expectedCount);
  }

  return {
    engine,
    endpoint,
    model: resolvedModel,
    embedTexts(texts) {
      const body = engine === "omlx" ? omlxTextRequestBody(resolvedModel, texts) : mlxServeTextRequestBody(resolvedModel, texts);
      return post(body, texts.length);
    },
    async embedImages(imageDataUrls) {
      if (engine === "omlx") {
        return post(omlxImageRequestBody(resolvedModel, imageDataUrls), imageDataUrls.length);
      }
      return mapWithConcurrency(imageDataUrls, mlxServeImageConcurrency, async (imageDataUrl) => {
        const [vector] = await post(mlxServeImageRequestBody(resolvedModel, imageDataUrl), 1);
        return vector;
      });
    },
  };
}
