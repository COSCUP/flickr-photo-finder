# ADR 0012: 將 Photo Finder 衍生並在地化為 COSCUP 專案

## 狀態

Accepted

## 背景

SITCON Flickr Photo Finder 建立了一套成熟的活動照片檢索、標記、Google Sheets 協作與 AI 候選審核機制。COSCUP 年會同樣擁有龐大的 Flickr 活動照片資產與籌備工作需求（社群宣傳、新聞稿、贊助結案佐證、活動回顧等）。

因此，專案自 SITCON 社群 fork 至 COSCUP 組織，需將系統在地化為 COSCUP 使用之系統，同時替換組織、Flickr 帳號（`coscup`）、專案 repo、品牌識別與 2026 年度官方贊助方案。

## 決策

Photo Finder 移植至 COSCUP 並完成在地化：

- **維持核心架構邊界**：延續 ADR 0001 至 0011 之決策，包含 Google Sheets-first、唯讀靜態前端、AI 候選約束（AI 僅標記 `ai_labeled`，由人工 review）、Flickr 來源不變，且**嚴禁人物身份索引**（ADR 0008）。
- **單一真理來源設定**：將 `config/project.json` 更新為 COSCUP 之設定（組織名稱 `COSCUP`、Flickr 路徑 `coscup`、Repo `COSCUP/flickr-photo-finder`、GA4 ID `G-C9EMTMDSS1`）。
- **贊助品項快照**：將 `data/sponsorship-items.json` 替換為 COSCUP 2026 官方贊助方案快照，涵蓋攤位、Keynote 廳垂吊布條、頸帶獨家贊助、會前派對、點心桌旗、迎賓牆、Job Wall、Main Track 與技術演講等，並對齊 `data/tag-taxonomy.json`。
- **品牌識別全面轉換**：更新前端 Web App（HTML title、Open Graph、Twitter Cards、標題）與 Google Apps Script 選單與側邊欄標題為 `COSCUP Photo Finder`。

## 取捨

優點：
- 完整承接 SITCON 成熟之治理機制、測試套件與 PWA 架構，無須重新造輪子。
- 贊助項目精準對齊 COSCUP 2026 現場曝光物資，能立即支援行銷與贊助組結案找圖需求。
- 專案維持 forkable 特性，未來若有新年度或跨社群需求可輕易套用。

代價：
- 既有 fixtures 照片資料暫為小樣本，正式上線時需以 COSCUP 專屬之 Google Sheets 綁定。
- 各年度相簿 catalog 需由 COSCUP Flickr 重新進行 discovery 與 intake。

## 相關文件

- `config/project.json`
- `data/sponsorship-items.json`
- `data/tag-taxonomy.json`
- `docs/adr/0004-sponsorship-items-snapshot.md`
- `docs/adr/0008-photo-index-product-boundary.md`
