# ADR 0013: 可選開啟的語意搜尋與以圖找圖

## 狀態

Accepted

## 背景

Finder 原本的搜尋是關鍵字比對：把查詢字串拆開，對照片的標題、描述、場景、用途等欄位做字串比對，再加上同義詞別名。找圖者實際輸入的常常是工作情境或畫面描述，例如「講者在台上和觀眾互動」，用詞和 metadata 不一致時就會整批落空。

PoC 評測的方式如下：

- 從索引抽樣 1,999 張照片，設計 24 個真實找圖題目，由 LLM 評審盲評前 12 張結果是否可用。評審與人工標記約有 73% 一致。
- 原本的關鍵字搜尋 P@12 為 15.3%，24 題中有 17 題前 12 張全部落空。
- EmbeddingGemma 2 文字向量 P@12 為 56.2%，圖片向量為 56.9%，Qwen3-VL-Embedding-2B 為 55.2%。
- EmbeddingGemma 2 文字向量加圖片向量，以 RRF 合併排名，在 256 維 8-bit 查詢時為 61.8%；採用的 512 維 int8、q4 查詢設定約為 64.9%。
- int8 與 float32 沒有品質差別；128 維與 1-bit 量化明顯變差。

評測樣本小、評審是模型，這些數字用來比較方案，不是對外承諾的品質指標。

專案本身是純靜態 GitHub Pages，沒有後端（見 [ADR 0002](0002-github-pages-artifact-deploy.md)）。語意搜尋必須在不新增伺服器、不引入 credential 的前提下提供。

## 決策

在公開 Finder 提供可選開啟的語意搜尋與以圖找圖，維持純靜態架構。

- **使用者主動開啟**：預設完全不下載任何向量、模型或函式庫。使用者在對話框中看到各項下載來源與大小後，才按下下載；下載有進度，可取消。
- **下載來源與大小**：
  - 照片向量來自同一個 Pages 部署，以 gzip 傳送，文字與圖片兩份合計約 27 MB（解壓後 35 MB）；只用以圖找圖時只下載圖片向量，約 13 MB。
  - 查詢文字模型的權重來自 huggingface.co，固定 revision，約 175 MB。tokenizer（`tokenizer.json`，32 MB）huggingface.co 不壓縮，所以把同一個檔案以 gzip 存進 repo（`data/embeddings/query-vocabulary.json.gz`），由本站提供，約 5.4 MB；build 與 `finder:check` 驗證它和固定 revision 的 sha256 相同，瀏覽器下載後也會驗證，不符時改向 huggingface.co 下載。
  - transformers.js 與 ONNX Runtime WASM 來自 jsDelivr，固定版本 4.3.2；jsDelivr 壓縮傳輸約 6 MB，解壓後約 27 MB。WASM 與模型同時下載並顯示進度，不等模型下載完才開始。
- **揭露使用的模型**：對話框列出照片向量與查詢使用的模型、量化方式、固定 revision 與授權（Google EmbeddingGemma 2，Apache 2.0），內容取自部署的 `vectors/meta.json`。
- **查詢不離開瀏覽器**：查詢文字只在瀏覽器內由 Web Worker 轉成向量，不送往任何第三方，也不送進 GA4。但模型與函式庫的下載請求會連到 huggingface.co 與 jsDelivr，這是使用者開啟前必須被告知的事實。
- **查詢模型的執行後端：WebGPU 優先，其餘用 WASM**：沒有可用的 WebGPU adapter，或 GPU 廠商是 `arm`（Mali）時，改用 ORT-web 的 WASM 後端在 CPU 執行。量化模型用到 WASM 後端沒有實作的 `GatherBlockQuantized`，worker 會在建立 session 前，把模型結構（約 0.5 MB）裡這個節點改寫成標準運算子，權重檔不變，結果與原運算子完全相同。實測每次查詢：Mac（Apple GPU、WebGPU）約 25 ms；Pixel 8 Pro（Mali）用 WebGPU 約 10 秒、用 WASM 不到 1 秒。
- **向量來源與格式**：由維護者在本機用 EmbeddingGemma 2 8-bit 產生文字向量與圖片向量，截斷到 512 維後以 int8 逐列量化。文字向量的輸入是相簿標題加 AI 視覺描述與場景、氛圍、用途標籤；圖片向量只使用 Flickr 縮圖。
- **向量以不可變分段 commit 進 repo，瀏覽器也以分段下載**：放在 `data/embeddings/`，只增量加入差異；同一張照片後面的分段覆蓋前面的。Pages build 把分段原樣（gzip）放進 artifact，瀏覽器依 photo_id 組成 `photos-index.json` 列順序的向量，並依 sha256 各自快取分段。新增一場活動（約 1,500 張）只需下載約 1.2 MB，AI 重新描述 2,000 張約 0.8 MB，不必重新下載整份約 27 MB 的向量；只有維護者執行 `--compact` 合併分段時，所有人才需要完整重新下載一次。
- **結果分頁**：開啟語意搜尋且有查詢時，結果分成「語意相近前 N 張」與「關鍵字命中」兩個分頁。關鍵字分頁完全維持原本行為；語意分頁套用其他篩選，但忽略排序選單。沒有向量的照片只會出現在關鍵字分頁。
- **以圖找圖**：在預覽對話框以單張照片的圖片向量排名相似照片，只用圖片向量，不需要載入文字模型。
- **沒有向量就隱藏入口**：`data/embeddings/` 不存在或沒有分段時，build 不產生向量，前端不顯示任何語意搜尋入口。

## 與 ADR 0007 的關係

語意搜尋的結果只是相似度排序，不是照片有或沒有的判定。

- 語意分頁固定顯示前 N 張，摘要必須說明「依語意相似度排序的前 N 張，不代表全部符合」。
- 排名最後一張也只是「比其他照片更相似」，不表示它符合需求；排名之外也不表示 Flickr 沒有相關照片。
- 缺少向量的照片不出現在語意分頁，因此語意分頁的空白或不足不能解讀成索引內沒有相關照片。
- 空結果仍須遵守 ADR 0007 的說法：沒有命中目前索引。

## 與 ADR 0008 的邊界

本功能使用的是一般圖片與文字 embedding 的整體視覺或語意相似度，不是人臉辨識，也不是人物分群。

- 不提供辨識、比對或追蹤特定人物的功能，不輸出 person id、人名或身份標記。
- 不建立人物索引：向量沒有針對臉部區域做處理，也不提供「這個人還出現在哪些照片」的操作。
- 以圖找圖找到的是構圖、場景與整體外觀相似的照片，不保證也不應宣稱是同一個人。
- 若未來要提供任何人物相關的搜尋，仍須依 ADR 0008 另開治理設計，不能延伸本功能。

## 取捨

優點：

- 在不新增後端、不保存 credential 的前提下，把原本找不到的描述型查詢大幅改善。
- 預設不下載，不增加一般使用者的流量；開啟前明示來源與大小。
- 向量分段不可變，git diff 只會新增檔案，補算新照片的成本與儲存量隨差異成長。
- 以圖找圖只需約 13 MB，不必下載文字模型。

代價：

- 第一次完整開啟約下載 213 MB，行動網路使用者需要自行判斷。
- 新增兩個第三方下載依賴（huggingface.co、jsDelivr）。它們不可用時語意搜尋就無法開啟，但關鍵字搜尋不受影響。版本與 revision 固定，避免上游內容在使用者不知情下改變。
- 不支援 WebGPU 或使用 Mali GPU 的環境以 WASM 執行查詢模型，比 WebGPU 慢。GitHub Pages 無法設定跨來源隔離所需的標頭，改由 service worker 替頁面與 worker 腳本加上 `Cross-Origin-Opener-Policy: same-origin` 與 `Cross-Origin-Embedder-Policy: credentialless`，WASM 因此可用 4 條執行緒（Pixel 8 Pro 每次查詢約 735 → 387 ms）。第一次造訪、service worker 尚未接管時，以及不支援 `credentialless` 的 Safari，仍是單執行緒。之後若要嵌入跨來源 iframe，需要加上 `credentialless` 屬性。
- 維護者需要一台能跑 EmbeddingGemma 2 的 Apple Silicon Mac；沒有 GPU 類加速時補算會很慢。
- 新照片在維護者補算向量之前沒有向量，只能被關鍵字搜尋找到。
- repo 會累積二進位分段檔，需要定期 compact 控制體積。
- 瀏覽器端使用 q4 量化模型查詢，和離線產生向量的 8-bit 模型不完全相同；品質數字已在 PoC 以同樣設定量測，但換模型時必須重新評估。

## 替代方案

- Qwen3-VL-Embedding-2B：文字與圖片共用同一個多模態空間，但查詢端需要在後端推論，不符合純靜態，且 PoC 品質沒有優於 EmbeddingGemma 2。
- 伺服器端查詢：下載量最小、品質可更高，但需要維運伺服器與 API 防濫用，和 ADR 0002 的唯讀靜態邊界衝突。
- 只用文字向量：不需要圖片向量，少約 18 MB，但失去以圖找圖，且混合 RRF 的品質高於單獨文字向量。
- CLIP 類模型：圖文共用空間、查詢模型小，但對中文描述與場景敘述的效果不在 PoC 範圍內，未採用。
- 預設開啟：體驗最順，但會對所有使用者強制下載數百 MB，已否決。

## 維護邊界與重新評估條件

維護邊界：

- 向量分段只由維護者在本機以 `pnpm finder:embed` 產生，不在 CI 與 Pages build 內執行模型。
- service worker 不處理 `data/finder-data/vectors/`，不得加入 precache；向量快取由前端自己的 Cache API 管理。
- 不得把查詢文字或查詢向量送往第三方或寫入 GA4。
- 新增任何和人物有關的向量用途前，先回到 ADR 0008。

出現以下情況時重新評估本決策：

- 有更小、品質相當的瀏覽器端文字模型，使首次下載量明顯下降。
- ORT-web 的 WASM 後端支援 `GatherBlockQuantized` 時，移除模型結構的改寫；其他 GPU 廠商在 WebGPU 上明顯比 WASM 慢時，加入 `wasmPreferredGpuVendors`。
- huggingface.co 或 jsDelivr 的可用性、條款或固定版本策略改變。
- 照片數量成長到向量檔案大小或瀏覽器記憶體成為瓶頸。
- GA4 或使用者回饋顯示開啟率過低，或下載量造成實際困擾。
- 專案取得可維運的後端，使伺服器端查詢成為合理選項。
- 換模型、換維度或換文件格式時，需要重新執行盲評並更新上述評測數字。

## 相關文件

- `docs/public-frontend-architecture.md`
- `docs/semantic-search-maintenance.md`
- `docs/adr/0002-github-pages-artifact-deploy.md`
- `docs/adr/0007-finder-index-results-are-not-absence-proof.md`
- `docs/adr/0008-photo-index-product-boundary.md`
