# 語意搜尋向量維護

這份文件給維護者，說明如何產生與更新語意搜尋用的照片向量。決策背景與取捨看 [ADR 0013](adr/0013-opt-in-semantic-search.md)，前端如何使用這些向量看 `docs/public-frontend-architecture.md`。

向量只由維護者在本機產生並 commit 進 `data/embeddings/`，不在 CI 或 Pages build 內執行模型。沒有補算時，新照片只是沒有向量，只能被關鍵字搜尋找到，Finder 其他功能不受影響。

## 什麼時候要跑

- 新增相簿、同步新照片進 Sheets 之後：新照片沒有圖片向量與文字向量。
- 大量修改描述或標籤之後：文字向量依描述與標籤內容算出，內容變了才會重算。
- 換模型或改維度時：走下方的重建流程。

## 前置需求

- Apple Silicon Mac。
- 預設使用本機的 mlx-serve：位址 `http://127.0.0.1:11234`，模型 `mlx-community/embeddinggemma-2-8bit`，不需要 API key。
- 也可改用 omlx（`--engine omlx`）：預設位址 `http://127.0.0.1:8000`，模型 `embeddinggemma-2-8bit`，API key 以環境變數 `FINDER_EMBED_API_KEY` 提供，不寫進任何檔案或指令列。
- 兩種引擎產生的向量 cosine 在 0.9997 以上，可以互換，換引擎不必重算。
- 不需要 Google 憑證：照片資料沿用 `finder:build` 的公開 CSV 讀取方式。

## 指令

先看會算多少，不產生任何檔案：

```sh
pnpm finder:embed --dry-run
```

只算差異並寫入新分段（預設 mlx-serve）：

```sh
pnpm finder:embed
```

改用 omlx：

```sh
export FINDER_EMBED_API_KEY=...   # 填入你的本機服務 API key
pnpm finder:embed --engine omlx
```

其他選項：

```sh
# 只處理其中一種向量（text 或 image）
pnpm finder:embed --kind image

# 合併成一個基礎分段，並移除 Sheets 已經不存在的照片
pnpm finder:embed --compact

# 換模型或改維度時全部重算
pnpm finder:embed --rebuild
```

引擎與位址用 `--engine mlx-serve|omlx`（預設 mlx-serve）、`--url` 指定，完整選項以 `pnpm finder:embed --help` 為準；`--help` 不需要連線任何服務。

縮圖會從 Flickr 下載 `_n.jpg`，快取在 `tmp/finder-embeddings/thumbs/`，這個目錄不 commit。

## 增量規則

- 圖片向量：只補還沒有向量的 photo_id。已有向量的照片不會因為描述改變而重算。
- 文字向量：以文件文字的 sha256 判斷，描述或標籤改變使文件文字不同時才重算；沒有改變就不算。
- 每次執行只新增分段，不修改既有分段。分段放在 `data/embeddings/segments/`，檔名含日期時間與類型，旁邊的 `.json` 記錄 photo_id（文字分段另有文字 hash）。
- `data/embeddings/index.json` 依序列出所有分段；同一類型、同一張照片，後面的分段覆蓋前面的。
- 沒有差異時不會新增分段。
- 分段累積變多、或有照片已從 Sheets 移除時，執行 `--compact` 合併成一個基礎分段。compact 會改寫 repo 內的檔案，commit 時會有大量二進位 diff，不要和一般補算混在同一個 commit。

## 換模型或改維度

1. 確認新模型的文件格式、查詢前綴與瀏覽器端可用的查詢模型是否改變；這些都記錄在 `index.json` 與 `vectors/meta.json`，前端依 `meta.json` 載入對應模型。
2. 換模型後，瀏覽器端查詢模型必須和產生照片向量的模型屬於同一個向量空間，否則排名沒有意義。先跑小範圍盲評，確認品質後再繼續，並更新 ADR 0013 的評測數字。
3. 用新的引擎設定執行 `pnpm finder:embed --rebuild`，所有照片全部重算。
4. 驗證後 commit；舊分段由重建取代。
5. 前端若需要改查詢模型的 id、revision 或 dtype，連同瀏覽器端程式一起修改，不要只改其中一邊。

## 驗證

產生分段後：

```sh
pnpm finder:build
pnpm finder:check
```

`finder:build` 會依 `photos-index.json` 的列順序輸出 `data/finder-data/vectors/`；`finder:check` 驗證 `meta.json`、檔案大小必須等於 `rowCount × 516`，以及 worker 存在。接著建議在本機預覽開啟語意搜尋，確認進度、查詢結果與以圖找圖都正常。

再跑一次 `pnpm finder:embed --dry-run`，沒有差異時應該顯示不需要補算。

## 常見問題

| 症狀 | 可能原因 | 處理 |
| --- | --- | --- |
| 連不上 embedding 服務 | mlx-serve 或 omlx 沒啟動、模型沒載入、位址不對 | 確認服務在執行，並用 `--engine`、`--url` 指定引擎與位址。 |
| 回應 401 或 403 | 使用 omlx 時 `FINDER_EMBED_API_KEY` 沒設定或不正確（mlx-serve 不需要 API key） | 重新設定環境變數；不要把 key 寫進檔案。 |
| mlx-serve 的 `/v1/models` 顯示 `hidden_size: 512`，擔心維度不對 | 那是文字主幹的寬度，實際輸出是 768 維 | 不需處理，工具會把輸出截斷成 512 維。 |
| 縮圖下載失敗 | Flickr 暫時限流或網路問題 | 重跑即可，已下載的縮圖會使用快取。 |
| 重跑後仍有差異 | 描述或標籤剛好又改變 | 先同步並重新讀取公開資料，再重跑。 |
| `finder:check` 說檔案大小不符 | 分段檔被手動修改或 build 中斷 | 重新執行 `pnpm finder:build`；仍失敗時檢查 `index.json` 的 sha256。 |
| 檔名被 check 擋下 | 檔名含 token、secret、credential、oauth 等字樣 | 分段檔名由工具產生，不要手動改名。 |
| 前端沒有語意搜尋入口 | `data/embeddings/` 沒有分段，build 沒有產生向量 | 先產生分段，再 build。 |
| 某些照片只出現在關鍵字分頁 | 該照片還沒有向量 | 執行 `pnpm finder:embed` 補算。 |

## 前端除錯

在網址加上 `?semantic-debug` 時，畫面左下角會顯示查詢模型的載入計時與每次查詢的耗時，並以 `./semantic-debug?kind=...` 請求送回服務該頁的主機（正式站會回 404，只在本機伺服器的 log 有用）。沒有這個參數時只會在 console 印一行計時，不影響畫面與網路。

| 參數 | 用途 |
| --- | --- |
| `semantic-debug` | 開啟除錯。計時欄位：`import`、`config`、`runtime`、`tokenizer`、`modelFiles`（模型檔下載或從快取讀完）、`session`（推論 session 建好）、`warmup`（第一次推論完成），另附 `backend`（`webgpu` 或 `wasm`）、GPU 型號與支援的功能。 |
| `semantic-placement` | 搭配 `semantic-debug`，回報 ONNX Runtime 把哪些節點分配到 CPU。 |
| `semantic-dtype=q4f16` | 搭配 `semantic-debug`，改用同一個模型的其他量化版本做比較。 |
| `semantic-backend=wasm` 或 `webgpu` | 搭配 `semantic-debug`，強制查詢模型使用 WASM 或 WebGPU，覆蓋自動判斷；可用來在有 WebGPU 的電腦上模擬不支援 WebGPU 的瀏覽器。 |

手機上要測文字語意搜尋，需要 HTTPS 或 localhost，否則瀏覽器不提供 WebGPU 與 Cache API。

要看每個節點的耗時，可以在實驗用的 worker 對 session 開 `enableProfiling: true`，推論後呼叫 `model.sessions.model.endProfiling()`。ONNX Runtime Web 沒有檔案系統，會把整份 profile JSON 逐行印到 worker 的 console；`cat` 為 `Api` 的事件是每個 WebGPU program 的耗時，`Node` 事件是 CPU 端時間（受瀏覽器計時精度限制，約 100 µs）。profile 會累積整個 session，分析時只取最後一次推論的事件。

## 執行後端與 Mali GPU

在 Pixel 8 Pro（Mali-G715）上，查詢模型用 WebGPU 執行時，一次查詢約 10 秒，其中 98% 以上都花在 146 個 q4 `MatMulNBits` 節點；模型其餘節點在 GPU 上合計約 0.13 秒。改成 DP4A kernel 或 q4f16 都沒有改善，GPU 在查詢期間維持在最高頻率附近，簡單的矩陣乘法測試也只有約 55 GFLOPS；只把 `MatMulNBits` 交給 CPU 也還要約 4 秒，因為每個節點都要在 GPU 與 CPU 之間搬資料。整個模型改用 WASM 執行後，每次查詢 0.7 到 0.9 秒、建立 session 約 2.3 秒、warmup 約 1.1 秒，所以 GPU 廠商是 `arm` 時直接使用 WASM。

頁面若能啟用跨來源隔離，WASM 可以使用多執行緒；但 GitHub Pages 無法設定 COOP/COEP 標頭，需要由 service worker 加上，影響整個網站，目前沒有採用。在 Pixel 8 Pro 上執行緒超過 4 條反而變慢，因為會排到效能較低的核心。

## 參考數字

mlx-serve 8-bit 在 M5 Max 上的實測：文字每秒約 365 至 390 筆（批次 16），圖片每秒約 17 張。mlx-serve 的圖片沒有批次，一次只能送一張，並發也不會加速。全部約 34k 張圖片約需 33 分鐘，文字約 1.5 分鐘。日常只補差異時通常在數分鐘內完成。換其他機器或引擎時速度會不同，只作為估算時間的參考。
