import assert from "node:assert/strict";
import { mkdtemp, readdir, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import {
  createEmbeddingClient,
  embeddingsEndpoint,
} from "../scripts/lib/finder/embedding-client.mjs";
import {
  buildEmbeddingSegment,
  buildVectorRows,
  compactEntries,
  computeEmbeddingDiff,
  countCoveredRows,
  dequantizeRow,
  documentText,
  embeddingRowBytes,
  mergeEmbeddingSegments,
  quantizeEmbedding,
  readEmbeddingStore,
  rowScale,
  sha256Hex,
  thumbnailUrl,
  truncateAndNormalize,
  writeEmbeddingStore,
} from "../scripts/lib/finder/embeddings.mjs";
import { buildFinderVectorPayloads } from "../scripts/lib/pages/finder-embeddings.mjs";

function seededVector(seed, length = 768) {
  let state = seed >>> 0;
  return Float32Array.from({ length }, () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 2 ** 32 - 0.5;
  });
}

function rowFor(seed) {
  return quantizeEmbedding(seededVector(seed));
}

function segment(kind, photoIds, seeds, textHashes = []) {
  return {
    kind,
    photoIds,
    textHashes,
    data: Buffer.concat(seeds.map(rowFor)),
  };
}

describe("finder embedding quantization", () => {
  it("round-trips a truncated, normalized vector within half a quantization step", () => {
    const raw = seededVector(7);
    const expected = truncateAndNormalize(raw);
    const row = quantizeEmbedding(raw);
    assert.equal(row.length, embeddingRowBytes);

    const scale = rowScale(row);
    const restored = dequantizeRow(row);
    let dot = 0;
    let norm = 0;
    for (let index = 0; index < expected.length; index += 1) {
      assert.ok(Math.abs(restored[index] - expected[index]) <= scale / 2 + 1e-7);
      dot += restored[index] * expected[index];
      norm += restored[index] * restored[index];
    }
    assert.ok(Math.abs(Math.sqrt(norm) - 1) < 0.01);
    assert.ok(dot / Math.sqrt(norm) > 0.999);
  });

  it("treats scale 0 as a missing row", () => {
    const data = Buffer.concat([rowFor(1), Buffer.alloc(embeddingRowBytes), rowFor(2)]);
    assert.equal(dequantizeRow(data, 1), null);
    assert.notEqual(dequantizeRow(data, 2), null);
    assert.equal(countCoveredRows(data), 2);
  });

  it("rejects vectors shorter than 512 dims or all zero", () => {
    assert.throws(() => quantizeEmbedding(new Float32Array(256)), /at least 512/);
    assert.throws(() => quantizeEmbedding(new Float32Array(768)), /zero vector/);
  });
});

describe("finder embedding documents", () => {
  it("formats eg2-photo-doc-v1 text and omits empty parts", () => {
    assert.equal(
      documentText({
        album_title: "COSCUP 2026 Day 1",
        visual_description: "講者在台上演講。",
        scene_tags: ["演講", "舞台"],
        mood_tags: ["專業"],
        recommended_uses: [],
      }),
      "title: COSCUP 2026 Day 1 | text: 講者在台上演講。 場景：演講、舞台。 氛圍：專業。",
    );
    assert.equal(
      documentText({ album_title: "", visual_description: "", scene_tags: [], mood_tags: [], recommended_uses: ["社群貼文"] }),
      "title: none | text: 用途：社群貼文。",
    );
  });

  it("maps image_preview_url _q thumbnails to _n", () => {
    assert.equal(
      thumbnailUrl({ image_preview_url: "https://live.staticflickr.com/65535/1_abc_q.jpg" }),
      "https://live.staticflickr.com/65535/1_abc_n.jpg",
    );
    assert.equal(thumbnailUrl({ image_preview_url: "" }), "");
  });
});

describe("finder embedding segments", () => {
  it("lets later segments override earlier ones for the same kind and photo_id", () => {
    const merged = mergeEmbeddingSegments([
      segment("image", ["p1", "p2"], [1, 2]),
      segment("text", ["p1"], [3], ["h-text"]),
      segment("image", ["p2"], [4]),
    ]);
    assert.deepEqual(merged.image.get("p1").row, rowFor(1));
    assert.deepEqual(merged.image.get("p2").row, rowFor(4));
    assert.deepEqual(merged.text.get("p1").row, rowFor(3));
    assert.equal(merged.text.get("p1").textHash, "h-text");
    assert.equal(merged.text.has("p2"), false);
  });

  it("computes image diffs by missing photo_id and text diffs by document sha256", () => {
    const photos = [
      { photo_id: "p1", image_preview_url: "https://example.test/p1_q.jpg", visual_description: "舞台" },
      { photo_id: "p2", image_preview_url: "https://example.test/p2_q.jpg", visual_description: "攤位" },
      { photo_id: "p3", image_preview_url: "", visual_description: "" },
      { photo_id: "p1", image_preview_url: "https://example.test/p1_q.jpg", visual_description: "重複列" },
    ];
    const unchangedHash = sha256Hex(documentText(photos[0]));
    const merged = mergeEmbeddingSegments([
      segment("image", ["p1"], [1]),
      segment("text", ["p1", "p2"], [2, 3], [unchangedHash, "stale-hash"]),
    ]);

    const image = computeEmbeddingDiff({ kind: "image", merged, photos });
    assert.deepEqual(image.pending, [{ photoId: "p2", url: "https://example.test/p2_n.jpg" }]);
    assert.equal(image.skipped, 1);

    const text = computeEmbeddingDiff({ kind: "text", merged, photos });
    assert.deepEqual(text.pending.map((item) => item.photoId), ["p2"]);
    assert.equal(text.pending[0].textHash, sha256Hex(documentText(photos[1])));
    assert.equal(text.skipped, 1);
  });

  it("writes, reads, and replaces segment files with index.json", async () => {
    const dir = await mkdtemp(join(tmpdir(), "finder-embeddings-test-"));
    try {
      const createdAt = new Date(2026, 10, 14, 21, 30);
      const first = buildEmbeddingSegment({
        kind: "text",
        entries: [{ photoId: "p1", row: rowFor(1), textHash: "h1" }],
        createdAt,
      });
      assert.equal(first.segment.file, "segments/20261114-2130-text.bin");
      let index = await writeEmbeddingStore({ embeddingsDir: dir, index: null, built: [first] });

      const second = buildEmbeddingSegment({
        kind: "text",
        entries: [{ photoId: "p2", row: rowFor(2), textHash: "h2" }],
        createdAt,
        existingFiles: new Set(index.segments.map((item) => item.file)),
      });
      assert.equal(second.segment.file, "segments/20261114-2130-text-2.bin");
      index = await writeEmbeddingStore({ embeddingsDir: dir, index, built: [second] });

      const store = await readEmbeddingStore(dir);
      assert.equal(store.segments.length, 2);
      assert.deepEqual(store.segments[1].textHashes, ["h2"]);
      const merged = mergeEmbeddingSegments(store.segments);
      assert.deepEqual(compactEntries({ kind: "text", merged, photoIds: ["p2", "gone"] }).map((entry) => entry.photoId), ["p2"]);

      const base = buildEmbeddingSegment({
        kind: "text",
        entries: compactEntries({ kind: "text", merged, photoIds: ["p1", "p2"] }),
        createdAt: new Date(2026, 10, 15, 8, 0),
      });
      index = await writeEmbeddingStore({ embeddingsDir: dir, index, built: [base], replaceKinds: ["text"] });
      assert.deepEqual(index.segments.map((item) => item.file), ["segments/20261115-0800-text.bin"]);
      assert.deepEqual((await readdir(join(dir, "segments"))).sort(), ["20261115-0800-text.bin", "20261115-0800-text.json"]);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it("returns no store when the embeddings directory does not exist", async () => {
    const store = await readEmbeddingStore(join(tmpdir(), "finder-embeddings-missing-dir"));
    assert.deepEqual(store, { index: null, segments: [] });
  });
});

describe("finder embedding Pages artifact", () => {
  it("orders rows by photos-index photo_id and zero-fills missing rows", () => {
    const merged = mergeEmbeddingSegments([segment("image", ["p3", "p1"], [3, 1])]);
    const { data, covered } = buildVectorRows({ kind: "image", merged, photoIds: ["p1", "p2", "p3"] });
    assert.equal(data.length, 3 * embeddingRowBytes);
    assert.equal(covered, 2);
    assert.deepEqual(data.subarray(0, embeddingRowBytes), rowFor(1));
    assert.ok(data.subarray(embeddingRowBytes, 2 * embeddingRowBytes).every((byte) => byte === 0));
    assert.equal(dequantizeRow(data, 1), null);
    assert.deepEqual(data.subarray(2 * embeddingRowBytes), rowFor(3));
  });

  it("builds meta.json with rowCount x 516 byte sets and skips kinds without segments", () => {
    const payloads = buildFinderVectorPayloads({
      segments: [segment("text", ["p2"], [2], ["h2"])],
      photoIds: ["p1", "p2", "p3", "p4"],
      photosSha256: "abc",
    });
    assert.equal(payloads.meta.rowCount, 4);
    assert.equal(payloads.meta.photosSha256, "abc");
    assert.deepEqual(Object.keys(payloads.meta.sets), ["text"]);
    const set = payloads.meta.sets.text;
    assert.equal(set.bytes, 4 * 516);
    assert.equal(payloads.files["text-int8.bin"].length, set.bytes);
    assert.equal(set.sha256, sha256Hex(payloads.files["text-int8.bin"]));
    assert.equal(set.covered, 1);
    assert.equal(buildFinderVectorPayloads({ segments: [], photoIds: ["p1"], photosSha256: "abc" }), null);
  });
});

describe("finder embedding client", () => {
  it("resolves the embeddings endpoint from a base URL", () => {
    assert.equal(embeddingsEndpoint("http://127.0.0.1:8000"), "http://127.0.0.1:8000/v1/embeddings");
    assert.equal(embeddingsEndpoint("http://127.0.0.1:8000/v1/"), "http://127.0.0.1:8000/v1/embeddings");
    assert.equal(embeddingsEndpoint("http://127.0.0.1:8000/v1/embeddings"), "http://127.0.0.1:8000/v1/embeddings");
  });

  it("sends omlx items with an optional bearer key and orders vectors by index", async () => {
    const requests = [];
    const client = createEmbeddingClient({
      engine: "omlx",
      apiKey: "local-key",
      fetchImpl: async (url, init) => {
        requests.push({ url, init, body: JSON.parse(init.body) });
        return {
          ok: true,
          json: async () => ({ data: [{ index: 1, embedding: [0, 1] }, { index: 0, embedding: [1, 0] }] }),
        };
      },
    });
    const vectors = await client.embedImages(["data:image/jpeg;base64,AA==", "data:image/jpeg;base64,AQ=="]);
    assert.deepEqual(requests[0].body.items, [{ image: "data:image/jpeg;base64,AA==" }, { image: "data:image/jpeg;base64,AQ==" }]);
    assert.equal(requests[0].body.encoding_format, "float");
    assert.equal(requests[0].init.headers.authorization, "Bearer local-key");
    assert.deepEqual(vectors.map((vector) => [...vector]), [[1, 0], [0, 1]]);
  });

  it("defaults to mlx-serve and sends a text batch as input", async () => {
    const requests = [];
    const client = createEmbeddingClient({
      fetchImpl: async (url, init) => {
        requests.push({ url, init, body: JSON.parse(init.body) });
        return { ok: true, json: async () => ({ data: [{ index: 0, embedding: [1] }, { index: 1, embedding: [2] }] }) };
      },
    });
    assert.equal(client.endpoint, "http://127.0.0.1:11234/v1/embeddings");
    await client.embedTexts(["a", "b"]);
    assert.deepEqual(requests[0].body, { model: "mlx-community/embeddinggemma-2-8bit", input: ["a", "b"] });
    assert.equal(requests[0].init.headers.authorization, undefined);
  });

  it("sends mlx-serve images one per request with at most two in flight", async () => {
    const bodies = [];
    let inFlight = 0;
    let maxInFlight = 0;
    const client = createEmbeddingClient({
      engine: "mlx-serve",
      fetchImpl: async (_url, init) => {
        const body = JSON.parse(init.body);
        bodies.push(body);
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((resolve) => setTimeout(resolve, 5));
        inFlight -= 1;
        const url = body.messages[0].content[0].image_url.url;
        return { ok: true, json: async () => ({ data: [{ index: 0, embedding: [url.length] }] }) };
      },
    });
    const images = ["data:image/jpeg;base64,A", "data:image/jpeg;base64,AB", "data:image/jpeg;base64,ABC"];
    const vectors = await client.embedImages(images);
    assert.equal(bodies.length, 3);
    assert.deepEqual(bodies[0].messages, [
      { role: "user", content: [{ type: "image_url", image_url: { url: images[0] } }] },
    ]);
    assert.equal(maxInFlight, 2);
    assert.deepEqual(vectors.map((vector) => vector[0]), images.map((image) => image.length));
  });
});
