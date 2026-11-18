export function renderActiveFilters({ elements, activeFilterEntries }) {
  const entries = activeFilterEntries();
  elements.activeFilters.replaceChildren();
  if (entries.length === 0) {
    const empty = document.createElement("span");
    empty.className = "filter-chip muted-chip";
    empty.textContent = "未套用條件";
    elements.activeFilters.append(empty);
    return;
  }

  for (const [key, label, value, rawValue = value] of entries) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip";
    chip.dataset.filterKey = key;
    chip.dataset.filterValue = rawValue;
    chip.textContent = `${label}: ${value} ×`;
    elements.activeFilters.append(chip);
  }
}

export function resultContextText({ photos, filtered, controls, activeTask, activeFilterEntries }) {
  if (photos.length === 0) {
    return "尚未載入照片";
  }
  if (filtered.length === 0) {
    return "沒找到不一定代表沒有照片；有些標籤尚未確認，可放寬條件或回相簿看看。";
  }

  const task = activeTask();
  const taskPrefix = task.id === "all" ? "全部照片" : `優先檢視「${task.label}」`;
  let sortText = task.id === "all" ? "以推薦排序" : `優先檢視適合「${task.label}」的照片`;
  if (controls.sort.value === "discover") {
    sortText = `以${taskPrefix}亂序探索排序，分散年份、活動、相簿與素材包來源`;
  } else if (controls.sort.value === "newest") {
    sortText = "以年份新到舊排序";
  } else if (controls.sort.value === "oldest") {
    sortText = "以年份舊到新排序";
  } else if (controls.sort.value === "people-desc") {
    sortText = "以人數多到少排序";
  } else if (controls.sort.value === "people-asc") {
    sortText = "以人數少到多排序";
  }

  const filterText = activeFilterEntries()
    .filter(([key]) => key !== "task")
    .map(([, label, value]) => `${label} ${value}`)
    .join(" / ");
  return `${sortText}，這些是目前命中的照片；有些標籤尚未人工確認。${filterText ? `已套用：${filterText}` : "未套用額外篩選。"}`;
}

// Semantic and similarity rankings are a bounded top-N by similarity, not a
// match set; the wording must say so (ADR 0007).
export function semanticResultText({ mode, count, sourceTitle = "", pending = false, activeFilterEntries }) {
  const filterText = activeFilterEntries()
    .filter(([key]) => key !== "task" && key !== "search" && key !== "similar")
    .map(([, label, value]) => `${label} ${value}`)
    .join(" / ");
  const filterSuffix = filterText ? `已套用：${filterText}` : "未套用額外篩選。";
  if (pending) {
    return {
      summary: "正在計算語意相似度",
      context: "查詢內容只在這個瀏覽器裡計算。",
    };
  }
  if (mode === "similar") {
    return {
      summary: `與「${sourceTitle}」畫面相似的前 ${count} 張，不代表全部符合`,
      context: `依圖片向量的整體畫面相似度排序，不辨識人物；會套用其他篩選，但不使用排序選單。${filterSuffix}`,
    };
  }
  return {
    summary: `依語意相似度排序的前 ${count} 張，不代表全部符合`,
    context: `語意分頁會套用其他篩選，但不使用排序選單；沒有向量的照片只會出現在關鍵字分頁。${filterSuffix}`,
  };
}

export function renderResultTabs({ elements, show, view, semanticCount, keywordCount, pending = false }) {
  elements.resultTabs.hidden = !show;
  if (!show) {
    return;
  }
  elements.semanticTab.textContent = pending ? "語意相近（計算中）" : `語意相近（前 ${semanticCount} 張）`;
  elements.keywordTab.textContent = `關鍵字命中（${keywordCount} 張）`;
  for (const [tab, tabView] of [
    [elements.semanticTab, "semantic"],
    [elements.keywordTab, "keyword"],
  ]) {
    const selected = tabView === view;
    tab.setAttribute("aria-selected", selected ? "true" : "false");
    tab.tabIndex = selected ? 0 : -1;
    tab.classList.toggle("is-active", selected);
  }
}

export const autoLoadMoreDistancePx = 640;

export function shouldAutoLoadMore({
  panel,
  visibleCount,
  filtered,
  viewportHeight,
  distancePx = autoLoadMoreDistancePx,
} = {}) {
  if (!panel || !Array.isArray(filtered)) {
    return false;
  }
  const renderedCount = Math.min(Number(visibleCount) || 0, filtered.length);
  const remaining = filtered.length - renderedCount;
  if (panel.hidden || remaining <= 0 || filtered.length === 0) {
    return false;
  }
  let height = viewportHeight;
  if (!Number.isFinite(height)) {
    height = typeof window === "undefined" ? 0 : window.innerHeight;
  }
  return panel.getBoundingClientRect().top <= height + distancePx;
}

export function updateTaskButtons({ elements, taskMode }) {
  for (const button of elements.taskModes.querySelectorAll(".task-mode")) {
    const selected = button.dataset.taskMode === taskMode;
    button.classList.toggle("is-active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

export function updateLoadMore({ elements, visibleCount, filtered }) {
  const renderedCount = Math.min(visibleCount, filtered.length);
  const remaining = filtered.length - renderedCount;
  elements.loadMorePanel.hidden = remaining <= 0 || filtered.length === 0;
  elements.loadMoreSummary.textContent = `已顯示 ${renderedCount} 張，尚有 ${remaining} 張`;
}

export function renderEmpty(grid, text) {
  const empty = document.createElement("div");
  empty.className = "empty";
  empty.textContent = text;
  grid.append(empty);
}
