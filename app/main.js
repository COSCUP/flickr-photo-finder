import {
  hasActiveFilters,
  hasTrackedResultState,
  resultCountBucket,
  resultsEventParams,
  sanitizeSearchTerm,
  scheduleResultsTracking as scheduleAnalyticsResultsTracking,
  setupAnalytics,
  trackEvent,
} from "./analytics.js";
import { aiAssistantEventParams, buildAiAssistantPrompt } from "./ai-assistant.js";
import {
  candidateCopyText,
  candidateLargeImageTargets,
  candidateOriginalPageUrls,
  renderCandidates,
  selectedPhotos,
} from "./candidates.js";
import { dataSources, projectConfigUrl } from "./config.js";
import {
  activeFilterEntries as buildActiveFilterEntries,
  applyControlsRegistry,
  bindControlDismissal,
  filterDefinitions,
  queryControls,
  queryElements,
  selectedControlValues,
  setControlValues,
  setupFilters,
  setupTaskModes,
  syncEnhancedSelects,
  updateFilterLayout,
} from "./controls.js";
import { loadFinderData, optionLabelsFor } from "./data-loader.js";
import { renderOverview as renderOverviewPanel } from "./overview-render.js";
import {
  copyTextToClipboard,
  displayImageUrl,
  downloadImageUrl,
  finderLink,
  imageDownloadFilename,
  largeImageUrl,
  originalSizePageUrl,
  photoAnchorId,
  photoTitle,
  renderPhotoDetails,
  renderPhotoCard,
  renderPhotoReference,
  renderPhotoStatuses,
  setTemporaryButtonText,
  sheetRowLink as buildSheetRowLink,
} from "./photo-render.js";
import { adjacentPhoto, horizontalSwipeStep } from "./preview-navigation.js";
import { registerPwa } from "./pwa.js";
import {
  renderActiveFilters,
  renderEmpty,
  renderResultTabs,
  resultContextText,
  semanticResultText,
  shouldAutoLoadMore,
  updateLoadMore,
  updateTaskButtons,
} from "./result-render.js";
import { applySearchRegistry, filterAndSortPhotos, matchesFilters } from "./search-sort.js";
import { createSemanticController } from "./semantic-controller.js";
import {
  eligibilityMask,
  enterSimilarState,
  exitSimilarState,
  hasVector,
  photosForRows,
  resolveResultView,
  semanticRanking,
  similarRanking,
} from "./semantic-search.js";
import { applyUrlStateRegistry, decodeUrlState, encodeUrlState, finderStateUrl } from "./url-state.js";
import {
  applyTaskModeRegistry,
  discoverCandidateLimit,
  discoverHistorySize,
  discoverWindowSize,
  pageSize,
  peopleCountFilters,
  resultTrackingDelayMs,
  searchDebounceMs,
  semanticRrfK,
  semanticTopK,
  taskModes,
} from "./task-modes.js";

const controls = queryControls();
const elements = queryElements();
// Semantic controls stay out of `controls` so generic filter binding skips them.
Object.assign(elements, {
  resultBar: document.querySelector(".result-bar"),
  resultTabs: document.querySelector("#resultTabs"),
  semanticTab: document.querySelector("#semanticTabButton"),
  keywordTab: document.querySelector("#keywordTabButton"),
  previewSimilar: document.querySelector("#previewSimilarButton"),
});
bindControlDismissal();

let photos = [];
let albums = [];
let photoSchema;
let currentResults = [];
let visibleCount = pageSize;
let renderTimer = 0;
let autoLoadMoreFrame = 0;
let projectConfig = {};
let optionLabelMaps = new Map();
let searchTokensForField = () => [];
let filterControlEventsBound = false;
let activePreviewPhoto = null;
let previewTrigger = null;
let previewTriggerPhotoId = "";
let loadPhotoDetails = async (photo) => photo;
let previewLoadToken = 0;
let previewSwipeState = null;
let suppressPreviewImageClick = false;
let sheetDragState = null;
let pwaRegistered = false;
let semantic = null;
let photoRowById = new Map();
let semanticMask = null;
let semanticMemo = { key: "", rows: [] };
let semanticPendingQuery = "";
let pendingSimilarPhotoId = "";
let currentView = { mode: "keyword", showTabs: false };
const failedSemanticQueries = new Set();

const state = {
  taskMode: "all",
  filters: Object.fromEntries(filterDefinitions.map((definition) => [definition.key, []])),
  selectedPhotoIds: new Set(),
  promotedPhotoIds: new Set(),
  lastTrackedZeroState: "",
  view: "",
  similarPhotoId: "",
  similarReturn: null,
};

let filterControls = {};

function refreshFilterControls() {
  filterControls = Object.fromEntries(filterDefinitions.map((definition) => [definition.key, controls[definition.control]]));
}

refreshFilterControls();

function cleanFilterValues(values) {
  const seen = new Set();
  return (Array.isArray(values) ? values : [values])
    .map((value) => String(value ?? "").trim())
    .filter(Boolean)
    .filter((value) => {
      const key = value.toLowerCase();
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

function setFilterValues(key, values, { syncControl = true } = {}) {
  if (!Object.hasOwn(state.filters, key)) {
    return;
  }
  state.filters[key] = cleanFilterValues(values);
  if (syncControl && filterControls[key]) {
    setControlValues(filterControls[key], state.filters[key], { dispatch: false });
  }
}

function clearFilterValue(key, value) {
  const target = String(value ?? "").trim().toLowerCase();
  if (!target) {
    setFilterValues(key, []);
    return;
  }
  setFilterValues(
    key,
    state.filters[key].filter((item) => item.toLowerCase() !== target),
  );
}

function syncControlsFromState() {
  for (const definition of filterDefinitions) {
    setControlValues(controls[definition.control], state.filters[definition.key] ?? [], { dispatch: false });
  }
  syncEnhancedSelects();
}

function syncStateFromControl(key) {
  const control = filterControls[key];
  if (!control) {
    return;
  }
  setFilterValues(key, selectedControlValues(control), { syncControl: false });
}

function photosSheetUrl() {
  const spreadsheetId = String(projectConfig.googleSheets?.spreadsheetId ?? "").trim();
  if (!spreadsheetId) {
    return "";
  }
  const gid = encodeURIComponent(String(projectConfig.googleSheets?.photosSheetGid ?? 0));
  return `https://docs.google.com/spreadsheets/d/${spreadsheetId}/edit?gid=${gid}#gid=${gid}`;
}

function setExternalLink(link, href) {
  if (!href) {
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    return;
  }
  link.href = href;
  link.removeAttribute("aria-disabled");
}

function updatePwaStatus({ text, hidden }) {
  elements.pwaStatus.textContent = text;
  elements.pwaStatus.hidden = hidden;
}

function maybeRegisterPwa(generatedAt = "") {
  if (pwaRegistered) {
    return;
  }
  pwaRegistered = true;
  registerPwa({
    generatedAt,
    onStatusChange: updatePwaStatus,
  });
}

function setModalOpen(open) {
  elements.modalBackdrop.hidden = !open || elements.photoPreviewDialog.open;
  document.body.classList.toggle("has-modal-open", open);
}

function closeFilterSheet() {
  resetSheetDragState();
  elements.searchPanel.classList.remove("is-filter-open");
  if (!elements.photoPreviewDialog.hidden || elements.sidePanel.classList.contains("is-candidate-open")) {
    return;
  }
  setModalOpen(false);
}

function openFilterSheet() {
  closeCandidateSheet();
  closePreview();
  elements.searchPanel.classList.add("is-filter-open");
  setModalOpen(true);
}

function closeCandidateSheet() {
  resetSheetDragState();
  elements.sidePanel.classList.remove("is-candidate-open");
  if (!elements.photoPreviewDialog.hidden || elements.searchPanel.classList.contains("is-filter-open")) {
    return;
  }
  setModalOpen(false);
}

function openCandidateSheet() {
  closeFilterSheet();
  closePreview();
  elements.sidePanel.classList.add("is-candidate-open");
  setModalOpen(true);
}

function closePreview() {
  const wasOpen = elements.photoPreviewDialog.open;
  previewLoadToken += 1;
  resetSheetDragState();
  elements.photoPreviewDialog.close();
  elements.photoPreviewDialog.hidden = true;
  activePreviewPhoto = null;
  if (!elements.searchPanel.classList.contains("is-filter-open") && !elements.sidePanel.classList.contains("is-candidate-open")) {
    setModalOpen(false);
  }
  if (wasOpen) {
    const trigger = previewTrigger?.isConnected && previewTrigger.getClientRects().length
      ? previewTrigger
      : document.getElementById(photoAnchorId(previewTriggerPhotoId))?.querySelector(".photo-link");
    trigger?.focus({ preventScroll: true });
    previewTrigger = null;
    previewTriggerPhotoId = "";
  }
}

function closeMobileOverlays() {
  elements.searchPanel.classList.remove("is-filter-open");
  elements.sidePanel.classList.remove("is-candidate-open");
  closePreview();
}

function isPreviewOpen() {
  return !elements.photoPreviewDialog.hidden && activePreviewPhoto;
}

function isMobileSheet() {
  return window.matchMedia("(max-width: 760px)").matches;
}

function scrollCandidateListToBottom() {
  if (isMobileSheet()) {
    return;
  }
  window.requestAnimationFrame(() => {
    elements.candidateList.scrollTop = elements.candidateList.scrollHeight;
  });
}

function resetSheetDragState() {
  if (sheetDragState?.sheet) {
    sheetDragState.sheet.style.transform = "";
  }
  sheetDragState = null;
}

function touchFromList(touches, identifier) {
  return Array.from(touches).find((touch) => touch.identifier === identifier) ?? null;
}

function shouldIgnoreSheetDragStart(event) {
  return Boolean(event.target.closest(".enhanced-select-panel, .candidate-list"));
}

function onSheetTouchStart(event, { sheet, isOpen, close }) {
  if (!isOpen() || !isMobileSheet() || event.touches.length !== 1 || shouldIgnoreSheetDragStart(event)) {
    return;
  }
  const touch = event.touches[0];
  sheetDragState = {
    sheet,
    close,
    touchId: touch.identifier,
    startY: touch.clientY,
    startScrollTop: sheet.scrollTop,
  };
}

function onSheetTouchMove(event) {
  if (!sheetDragState) {
    return;
  }
  const touch = touchFromList(event.touches, sheetDragState.touchId);
  if (!touch) {
    return;
  }
  const deltaY = touch.clientY - sheetDragState.startY;
  if (sheetDragState.startScrollTop > 0 || deltaY <= 0) {
    return;
  }
  event.preventDefault();
  const dragOffset = Math.min(deltaY, 160);
  sheetDragState.sheet.style.transform = `translateY(${Math.round(dragOffset)}px)`;
}

function onSheetTouchEnd(event) {
  if (!sheetDragState) {
    return;
  }
  const touch = touchFromList(event.changedTouches, sheetDragState.touchId);
  if (!touch) {
    resetSheetDragState();
    return;
  }
  const deltaY = touch.clientY - sheetDragState.startY;
  const shouldClose = sheetDragState.startScrollTop <= 0 && deltaY >= 96;
  const close = sheetDragState.close;
  resetSheetDragState();
  if (shouldClose) {
    close();
  }
}

function currentFilterSnapshot() {
  const filterCounts = Object.fromEntries(Object.entries(state.filters).map(([key, values]) => [`${key}Count`, values.length]));
  return {
    taskMode: state.taskMode,
    searchTerm: sanitizeSearchTerm(controls.search.value),
    recommendedUse: state.filters.use,
    mood: state.filters.mood,
    sortMode: controls.sort.value,
    album: state.filters.album,
    scene: state.filters.scene,
    peopleCount: state.filters.peopleCount,
    subjectType: state.filters.subjectType,
    orientation: state.filters.orientation,
    negativeSpace: state.filters.negativeSpace,
    safeCrop: state.filters.safeCrop,
    sponsorshipTag: state.filters.sponsorshipTag,
    sponsorshipItemCount: state.filters.sponsorshipItem.length,
    publicUseStatus: state.filters.publicStatus,
    priorityLevel: state.filters.priority,
    curationStatus: state.filters.curationStatus,
    collection: state.filters.collection,
    ...filterCounts,
    searchMode: currentView.mode,
    resultCount: currentResults.length,
  };
}

function scheduleResultsTracking(source) {
  scheduleAnalyticsResultsTracking(source, {
    getSnapshot: currentFilterSnapshot,
    delayMs: resultTrackingDelayMs,
  });
}

function applyProjectConfig(config) {
  projectConfig = config;
  const title = config.frontend?.appTitle ?? "Flickr Photo Finder";
  document.title = title;
  elements.appTitle.textContent = title;
  elements.sourceLink.href = config.flickr?.profileUrl ?? "https://www.flickr.com/";
  elements.sourceLink.textContent = config.frontend?.sourceLinkLabel ?? "Flickr";
  const repositoryUrl = String(config.repository?.url ?? "").trim();
  elements.repositoryLink.hidden = !repositoryUrl;
  if (repositoryUrl) {
    elements.repositoryLink.href = repositoryUrl;
    elements.repositoryLink.textContent = config.frontend?.repositoryLinkLabel ?? "GitHub 專案";
    elements.repositoryLink.title = config.frontend?.repositoryLinkTitle ?? "了解專案細節或回報問題";
  }
  setExternalLink(elements.aiAssistantSheetLink, photosSheetUrl());
  setupAnalytics(config);
}

function uniqueSorted(values) {
  return [...new Set(values.filter(Boolean))].sort((left, right) =>
    left.localeCompare(right, "zh-Hant-TW"),
  );
}

function optionLabels(fieldName) {
  return optionLabelsFor(optionLabelMaps, fieldName);
}

function labelFor(fieldName, value) {
  return optionLabels(fieldName).get(value) ?? value;
}

function activeTask() {
  return taskModes.find((task) => task.id === state.taskMode) ?? taskModes[0];
}

function updateTaskModeSummary() {
  elements.taskModeSummary.textContent = activeTask().label;
}

function initializeMobileTaskModePanel() {
  const mobileViewport = window.matchMedia("(max-width: 760px)");
  const syncTaskPanel = () => {
    elements.taskModeDetails.open = !mobileViewport.matches;
  };
  syncTaskPanel();
  mobileViewport.addEventListener("change", syncTaskPanel);
}

function currentPhotoFilters() {
  return Object.fromEntries([
    ["search", controls.search.value],
    ...filterDefinitions.map((definition) => [definition.filterParam ?? definition.key, state.filters[definition.key] ?? []]),
  ]);
}

function filteredAndSortedPhotos() {
  return filterAndSortPhotos(photos, {
    filters: currentPhotoFilters(),
    sortMode: controls.sort.value,
    task: activeTask(),
    discoverCandidateLimit,
    discoverHistorySize,
    discoverWindowSize,
    selectedPhotoIds: state.promotedPhotoIds,
  });
}

// Semantic views apply every filter except the keyword search text. The mask
// is built lazily so cached rankings skip the per-photo filter pass.
function semanticEligibility() {
  const filters = { ...currentPhotoFilters(), search: "" };
  return {
    key: JSON.stringify(filters),
    mask: () => {
      semanticMask = eligibilityMask(photos, (photo) => matchesFilters(photo, filters), semanticMask?.length === photos.length ? semanticMask : undefined);
      return semanticMask;
    },
  };
}

function memoizedRows(key, compute) {
  if (semanticMemo.key !== key) {
    semanticMemo = { key, rows: compute() };
  }
  return semanticMemo.rows;
}

function similarSourcePhoto() {
  const row = photoRowById.get(state.similarPhotoId);
  return row === undefined ? null : photos[row];
}

function requestQueryEmbedding(query) {
  if (semanticPendingQuery === query) {
    return;
  }
  semanticPendingQuery = query;
  semantic.embed(query).then(
    () => {
      if (semanticPendingQuery === query) {
        semanticPendingQuery = "";
      }
      if (controls.search.value.trim() === query) {
        render({ preservePage: true, source: "search" });
      }
    },
    () => {
      if (semanticPendingQuery === query) {
        semanticPendingQuery = "";
      }
      failedSemanticQueries.add(query);
      if (controls.search.value.trim() === query) {
        render({ preservePage: true, source: "search" });
      }
    },
  );
}

// Chooses the displayed list: keyword results stay the original behavior;
// semantic and similarity views are bounded top-N rankings over photos that
// pass every other filter.
function displayedResults(keywordResults) {
  const query = controls.search.value.trim();
  const view = resolveResultView({
    textReady: Boolean(semantic?.textReady()) && !failedSemanticQueries.has(query),
    imageReady: Boolean(semantic?.imageReady()),
    query,
    similarPhotoId: state.similarPhotoId,
    requestedView: state.view,
  });
  if (view.mode === "similar") {
    const sourcePhoto = similarSourcePhoto();
    const sourceRow = sourcePhoto ? photoRowById.get(sourcePhoto.photo_id) : -1;
    const image = semantic.sets.image;
    if (!sourcePhoto || !hasVector(image, sourceRow)) {
      return { ...view, results: [], sourcePhoto, missingSource: true };
    }
    const eligibility = semanticEligibility();
    const rows = memoizedRows(`similar:${sourceRow}:${eligibility.key}`, () =>
      similarRanking({ image, sourceRow, eligible: eligibility.mask(), topK: semanticTopK }),
    );
    return { ...view, results: photosForRows(photos, rows), sourcePhoto };
  }
  if (!view.showTabs) {
    return { ...view, results: keywordResults };
  }
  const vector = semantic.cachedEmbedding(query);
  if (!vector) {
    requestQueryEmbedding(query);
    return { ...view, results: view.mode === "semantic" ? [] : keywordResults, pending: true, semanticCount: 0 };
  }
  const eligibility = semanticEligibility();
  const sets = { text: semantic.sets.text, image: semantic.sets.image };
  const rows = memoizedRows(`query:${query}:${eligibility.key}`, () =>
    semanticRanking({ sets, query: vector, eligible: eligibility.mask(), topK: semanticTopK, rrfK: semanticRrfK }),
  );
  const semanticResults = photosForRows(photos, rows);
  return {
    ...view,
    results: view.mode === "semantic" ? semanticResults : keywordResults,
    semanticCount: semanticResults.length,
  };
}

function updatePreviewSimilarButton() {
  elements.previewSimilar.hidden = !semantic?.available || !semantic.hasImageVectors();
}

function scrollToResults() {
  elements.resultBar?.scrollIntoView({ block: "start" });
}

function similarStateSnapshot() {
  return {
    search: controls.search.value,
    view: state.view,
    similar: state.similarPhotoId,
    similarReturn: state.similarReturn,
  };
}

function applySimilarState(next) {
  controls.search.value = next.search;
  state.view = next.view;
  state.similarPhotoId = next.similar;
  state.similarReturn = next.similarReturn;
}

// Entering "similar to" gets its own history entry, so the browser back
// button returns to the previous query; removing the chip restores it too.
function applySimilarPhoto(photoId) {
  pendingSimilarPhotoId = "";
  closePreview();
  syncUrlState();
  window.history.pushState(null, "", window.location.href);
  applySimilarState(enterSimilarState(similarStateSnapshot(), photoId));
  render({ resetPage: true, source: "similar" });
  trackEvent("find_similar_photos", { photo_id: state.similarPhotoId, result_count: currentResults.length });
  scrollToResults();
}

function findSimilarToPreview() {
  if (!activePreviewPhoto || !semantic?.available) {
    return;
  }
  if (semantic.imageReady()) {
    applySimilarPhoto(activePreviewPhoto.photo_id);
    return;
  }
  pendingSimilarPhotoId = activePreviewPhoto.photo_id;
  closePreview();
  semantic.openDialog({ preferImage: true });
}

function onSemanticChange() {
  updatePreviewSimilarButton();
  if (pendingSimilarPhotoId && semantic.imageReady()) {
    applySimilarPhoto(pendingSimilarPhotoId);
    return;
  }
  semanticMemo = { key: "", rows: [] };
  render({ preservePage: true, preserveScroll: true });
}

function renderOverview() {
  renderOverviewPanel({ photos, photoSchema, elements, optionLabels });
}

function sheetRowLink(photo) {
  return buildSheetRowLink(photo, projectConfig);
}

async function photoWithDetails(photo) {
  try {
    return await loadPhotoDetails(photo);
  } catch {
    return photo;
  }
}

async function openPreview(photo) {
  if (!isPreviewOpen()) {
    previewTrigger = document.activeElement === document.body ? null : document.activeElement;
    previewTriggerPhotoId = photo.photo_id;
  }
  closeFilterSheet();
  closeCandidateSheet();
  const token = previewLoadToken + 1;
  previewLoadToken = token;
  const detailedPhoto = await photoWithDetails(photo);
  if (token !== previewLoadToken) {
    return;
  }
  activePreviewPhoto = detailedPhoto;
  const largeUrl = largeImageUrl(detailedPhoto);
  const originalUrl = originalSizePageUrl(detailedPhoto);
  const previewUrl = largeUrl || detailedPhoto.image_preview_url;
  elements.previewTitle.textContent = photoTitle(detailedPhoto);
  elements.previewMeta.textContent = [detailedPhoto.event_year, detailedPhoto.album_title].filter(Boolean).join(" / ");
  elements.previewImage.src = previewUrl;
  elements.previewImage.alt = [photoTitle(detailedPhoto), detailedPhoto.event_year].filter(Boolean).join(" ");
  renderPhotoStatuses(elements.previewStatuses, detailedPhoto, labelFor);
  renderPhotoReference(elements.previewReference, detailedPhoto);
  renderPhotoDetails(elements.previewDetails, detailedPhoto, { labelFor });
  controls.previewLarge.disabled = !largeUrl;
  controls.previewLarge.dataset.largeImageUrl = largeUrl;
  controls.previewCopyFlickr.disabled = !detailedPhoto.photo_url;
  setExternalLink(elements.previewImageLink, detailedPhoto.photo_url);
  elements.previewImageLink.title = detailedPhoto.photo_url ? "開啟 Flickr 照片頁" : "";
  setExternalLink(elements.previewOriginalLink, originalUrl);
  setExternalLink(elements.previewSheetLink, sheetRowLink(detailedPhoto));
  controls.previewCopyFlickr.title = "複製 Flickr 原始照片頁連結";
  controls.previewCopyFinder.title = "複製 Finder 中這張照片的 deep link";
  updatePreviewCandidateButton();
  updatePreviewSimilarButton();
  elements.photoPreviewDialog.hidden = false;
  if (!elements.photoPreviewDialog.open) {
    elements.photoPreviewDialog.showModal();
  }
  setModalOpen(true);
  controls.closePreview.focus({ preventScroll: true });
}

function navigatePreview(step) {
  if (!isPreviewOpen()) {
    return false;
  }
  const nextPhoto = adjacentPhoto(currentResults, activePreviewPhoto.photo_id, step);
  if (!nextPhoto) {
    return false;
  }
  openPreview(nextPhoto);
  return true;
}

function updatePreviewCandidateButton() {
  if (!activePreviewPhoto) {
    return;
  }
  const selected = state.selectedPhotoIds.has(activePreviewPhoto.photo_id);
  controls.previewCandidate.textContent = selected ? "已加入候選" : "加入候選";
  controls.previewCandidate.setAttribute("aria-pressed", selected ? "true" : "false");
  controls.previewCandidate.classList.toggle("is-selected", selected);
}

function currentUrlState() {
  return {
    taskMode: state.taskMode,
    search: controls.search.value,
    sort: controls.sort.value,
    filters: state.filters,
    selectedPhotoIds: state.selectedPhotoIds,
    view: state.view,
    similar: state.similarPhotoId,
  };
}

function candidateListLink() {
  return finderStateUrl(window.location.href, currentUrlState());
}

function activeFilterCount() {
  return displayedFilterEntries().filter(([key]) => key !== "task" && key !== "search").length;
}

function updateMobileSummary() {
  const selectedCount = state.selectedPhotoIds.size;
  controls.mobileFilter.textContent = activeFilterCount() > 0 ? `篩選 ${activeFilterCount()}` : "篩選";
  controls.mobileCandidate.textContent = `候選 ${selectedCount}`;
  elements.selectedNotice.hidden = selectedCount === 0;
  elements.selectedNotice.textContent = selectedCount === 0 ? "" : `目前已有 ${selectedCount} 張候選照片，可從「候選 ${selectedCount}」查看。`;
}

function renderPhoto(photo, resultRank, resultCount) {
  return renderPhotoCard(photo, resultRank, resultCount, {
    template: elements.template,
    selectedPhotoIds: state.selectedPhotoIds,
    projectConfig,
    labelFor,
    task: activeTask(),
    searchValue: controls.search.value,
    taskMode: state.taskMode,
    sortMode: controls.sort.value,
    toggleCandidate,
    openPreview,
    trackEvent,
  });
}

function loadMoreResults(source = "load_more") {
  if (visibleCount >= currentResults.length) {
    return false;
  }
  visibleCount = Math.min(visibleCount + pageSize, currentResults.length);
  render({ preservePage: true, source });
  trackEvent("load_more_results", {
    result_count: currentResults.length,
    visible_count: Math.min(visibleCount, currentResults.length),
    result_count_bucket: resultCountBucket(currentResults.length),
    task_mode: state.taskMode,
    sort_mode: controls.sort.value,
  });
  return true;
}

function isLoadMoreNearViewport() {
  return shouldAutoLoadMore({
    panel: elements.loadMorePanel,
    visibleCount,
    filtered: currentResults,
  });
}

function queueAutoLoadMore() {
  if (autoLoadMoreFrame || !isLoadMoreNearViewport()) {
    return;
  }
  autoLoadMoreFrame = window.requestAnimationFrame(() => {
    autoLoadMoreFrame = 0;
    if (isLoadMoreNearViewport()) {
      loadMoreResults("auto_load_more");
    }
  });
}

function toggleCandidate(photoId) {
  const isAdding = !state.selectedPhotoIds.has(photoId);
  if (!isAdding) {
    state.selectedPhotoIds.delete(photoId);
    state.promotedPhotoIds.delete(photoId);
    trackEvent("remove_candidate", { photo_id: photoId, task_mode: state.taskMode, sort_mode: controls.sort.value });
  } else {
    state.selectedPhotoIds.add(photoId);
    trackEvent("add_candidate", { photo_id: photoId, task_mode: state.taskMode, sort_mode: controls.sort.value });
  }
  updatePreviewCandidateButton();
  render({ preservePage: true, preserveScroll: true, source: "candidate" });
  if (isAdding) {
    scrollCandidateListToBottom();
  }
}

async function copyCandidateLink() {
  try {
    const copied = await copyTextToClipboard(candidateListLink());
    if (copied) {
      setTemporaryButtonText(controls.copyCandidates, "已複製");
      trackEvent("copy_candidate_list", {
        candidate_count: state.selectedPhotoIds.size,
        template_id: "finder_url",
        task_mode: state.taskMode,
        sort_mode: controls.sort.value,
      });
    }
  } catch {
    setTemporaryButtonText(controls.copyCandidates, "複製失敗");
  }
}

function setCandidateCopyMenuOpen(open) {
  controls.candidateCopyMenu.hidden = !open;
  controls.candidateCopyMenuButton.setAttribute("aria-expanded", open ? "true" : "false");
}

function closeCandidateCopyMenu() {
  setCandidateCopyMenuOpen(false);
}

function toggleCandidateCopyMenu() {
  if (controls.candidateCopyMenuButton.disabled) {
    return;
  }
  setCandidateCopyMenuOpen(controls.candidateCopyMenu.hidden);
}

async function copyCandidateTemplate(templateId) {
  const candidates = await Promise.all(selectedPhotos(state.selectedPhotoIds, photos).map(photoWithDetails));
  const text = candidateCopyText(
    candidates,
    { photoTitle, finderLink, candidateListLink, sheetRowLink, labelFor },
    templateId,
  );
  if (!text) {
    return;
  }
  try {
    const copied = await copyTextToClipboard(text);
    if (copied) {
      setTemporaryButtonText(controls.copyCandidates, "已複製");
      trackEvent("copy_candidate_list", {
        candidate_count: state.selectedPhotoIds.size,
        template_id: templateId,
        task_mode: state.taskMode,
        sort_mode: controls.sort.value,
      });
    }
  } catch {
    setTemporaryButtonText(controls.copyCandidates, "複製失敗");
  }
}

async function downloadCandidateLargeImages() {
  const candidates = selectedPhotos(state.selectedPhotoIds, photos);
  const targets = candidateLargeImageTargets(candidates, { largeImageUrl, imageDownloadFilename });
  if (targets.length === 0) {
    setTemporaryButtonText(controls.copyCandidates, "無大圖");
    return;
  }

  const originalText = controls.copyCandidates.textContent;
  let failedCount = 0;
  controls.copyCandidates.disabled = true;
  controls.candidateCopyMenuButton.disabled = true;
  controls.copyCandidates.textContent = "下載中";
  for (const target of targets) {
    try {
      await downloadImageUrl(target.url, target.filename);
    } catch {
      failedCount += 1;
    }
  }
  controls.copyCandidates.textContent = failedCount > 0 ? "部分失敗" : "已下載";
  trackEvent("download_candidate_large_images", {
    candidate_count: candidates.length,
    download_count: targets.length - failedCount,
    failed_count: failedCount,
    task_mode: state.taskMode,
    sort_mode: controls.sort.value,
  });
  window.setTimeout(() => {
    controls.copyCandidates.disabled = state.selectedPhotoIds.size === 0;
    controls.candidateCopyMenuButton.disabled = state.selectedPhotoIds.size === 0;
    controls.copyCandidates.textContent = originalText;
  }, 1900);
}

async function openCandidateOriginalPages() {
  const candidates = selectedPhotos(state.selectedPhotoIds, photos);
  const urls = candidateOriginalPageUrls(candidates, { originalSizePageUrl });
  if (urls.length === 0) {
    setTemporaryButtonText(controls.copyCandidates, "無原圖頁");
    return;
  }

  let openedCount = 0;
  for (const url of urls) {
    const opened = window.open(url, "_blank");
    if (opened) {
      opened.opener = null;
      openedCount += 1;
    }
  }

  const blockedCount = urls.length - openedCount;
  if (blockedCount === 0) {
    setTemporaryButtonText(controls.copyCandidates, "已打開");
  } else {
    try {
      const copied = await copyTextToClipboard(urls.join("\n"));
      setTemporaryButtonText(controls.copyCandidates, copied ? "已複製原圖頁" : "部分被阻擋");
    } catch {
      setTemporaryButtonText(controls.copyCandidates, "部分被阻擋");
    }
  }

  trackEvent("open_candidate_original_pages", {
    candidate_count: candidates.length,
    opened_count: openedCount,
    blocked_count: blockedCount,
    task_mode: state.taskMode,
    sort_mode: controls.sort.value,
  });
}

function runCandidateBatchAction(actionId) {
  if (actionId === "download_large") {
    downloadCandidateLargeImages();
    return;
  }
  if (actionId === "open_original_pages") {
    openCandidateOriginalPages();
  }
}

async function downloadPreviewLargeImage() {
  if (!activePreviewPhoto) {
    return;
  }
  const largeUrl = controls.previewLarge.dataset.largeImageUrl || largeImageUrl(activePreviewPhoto);
  if (!largeUrl) {
    return;
  }
  const originalText = controls.previewLarge.textContent;
  try {
    controls.previewLarge.disabled = true;
    controls.previewLarge.textContent = "下載中";
    await downloadImageUrl(largeUrl, imageDownloadFilename(activePreviewPhoto, largeUrl));
    controls.previewLarge.textContent = "已下載";
    trackEvent("download_preview_large_image", { photo_id: activePreviewPhoto.photo_id });
  } catch {
    controls.previewLarge.textContent = "下載失敗";
  } finally {
    window.setTimeout(() => {
      controls.previewLarge.disabled = false;
      controls.previewLarge.textContent = originalText;
    }, 1900);
  }
}

async function copyPreviewFlickrLink() {
  if (!activePreviewPhoto?.photo_url) {
    return;
  }
  try {
    const copied = await copyTextToClipboard(activePreviewPhoto.photo_url);
    if (copied) {
      setTemporaryButtonText(controls.previewCopyFlickr, "已複製");
      trackEvent("copy_flickr_link", { photo_id: activePreviewPhoto.photo_id });
    }
  } catch {
    setTemporaryButtonText(controls.previewCopyFlickr, "複製失敗");
  }
}

async function copyPreviewFinderLink() {
  if (!activePreviewPhoto) {
    return;
  }
  try {
    const copied = await copyTextToClipboard(finderLink(activePreviewPhoto));
    if (copied) {
      setTemporaryButtonText(controls.previewCopyFinder, "已複製");
      trackEvent("copy_finder_link", { photo_id: activePreviewPhoto.photo_id });
    }
  } catch {
    setTemporaryButtonText(controls.previewCopyFinder, "複製失敗");
  }
}

function clearCandidates() {
  state.selectedPhotoIds.clear();
  state.promotedPhotoIds.clear();
  render({ preservePage: true, preserveScroll: true, source: "candidate" });
}

function currentAiAssistantPrompt() {
  return buildAiAssistantPrompt({
    sheetUrl: photosSheetUrl(),
    taskLabel: activeTask().label,
    searchValue: controls.search.value,
    filterEntries: activeFilterEntries(),
  });
}

function currentAiAssistantEventParams() {
  return aiAssistantEventParams({
    taskMode: state.taskMode,
    searchValue: controls.search.value,
    filterEntries: activeFilterEntries(),
  });
}

async function copyAiAssistantPrompt() {
  try {
    const copied = await copyTextToClipboard(currentAiAssistantPrompt());
    if (copied) {
      setTemporaryButtonText(controls.copyAiAssistantPrompt, "已複製");
      trackEvent("copy_ai_assistant_prompt", currentAiAssistantEventParams());
    }
  } catch {
    setTemporaryButtonText(controls.copyAiAssistantPrompt, "複製失敗");
  }
}

function activeFilterEntries() {
  return buildActiveFilterEntries({ state, controls, activeTask: activeTask() });
}

// Chips also show the removable "similar to" condition. It stays out of
// activeFilterEntries so AI assistant prompts only describe keyword filters.
function displayedFilterEntries() {
  const entries = activeFilterEntries();
  if (state.similarPhotoId) {
    const sourcePhoto = similarSourcePhoto();
    const label = semantic?.imageReady() ? "相似於" : "相似於（需開啟以圖找圖）";
    entries.push(["similar", label, sourcePhoto ? photoTitle(sourcePhoto) : state.similarPhotoId, state.similarPhotoId]);
  }
  return entries;
}

function updateResetButtonState() {
  const hasResettableState = displayedFilterEntries().length > 0 || controls.sort.value !== "recommended";
  controls.reset.classList.toggle("is-active", hasResettableState);
}

function clearFilter(key, value = "") {
  if (key === "task") {
    state.taskMode = "all";
  } else if (key === "similar") {
    applySimilarState(exitSimilarState(similarStateSnapshot()));
  } else if (key === "search") {
    controls.search.value = "";
  } else {
    clearFilterValue(key, value);
  }
  syncEnhancedSelects();
  render({ resetPage: true, source: "filter" });
}

function render({ resetPage = false, preservePage = false, preserveScroll = false, source = "" } = {}) {
  const scrollX = window.scrollX;
  const scrollY = window.scrollY;
  const restoreScroll = () => {
    if (preserveScroll) {
      window.requestAnimationFrame(() => window.scrollTo(scrollX, scrollY));
    }
  };
  const keywordResults = filteredAndSortedPhotos();
  const display = displayedResults(keywordResults);
  const filtered = display.results;
  currentView = { mode: display.mode, showTabs: display.showTabs };
  currentResults = filtered;
  if (resetPage || (!preservePage && visibleCount <= 0)) {
    visibleCount = pageSize;
  }

  updateTaskButtons({ elements, taskMode: state.taskMode });
  updateTaskModeSummary();
  updateFilterLayout({ controls, elements, taskMode: state.taskMode });
  renderActiveFilters({ elements, activeFilterEntries: displayedFilterEntries });
  updateResetButtonState();
  renderResultTabs({
    elements,
    show: display.showTabs,
    view: display.mode,
    semanticCount: display.semanticCount ?? 0,
    keywordCount: keywordResults.length,
    pending: Boolean(display.pending),
  });
  renderCandidates({
    selectedPhotoIds: state.selectedPhotoIds,
    photos,
    elements,
    controls,
    photoTitle,
    finderLink,
    labelFor,
    toggleCandidate,
    openPreview,
    displayImageUrl,
  });
  updateMobileSummary();
  elements.grid.replaceChildren();
  elements.grid.setAttribute("aria-busy", "false");
  if (display.mode === "keyword") {
    elements.summary.textContent = `${filtered.length} / ${photos.length} 張照片`;
    elements.context.textContent = resultContextText({ photos, filtered, controls, activeTask, activeFilterEntries });
  } else {
    const text = semanticResultText({
      mode: display.mode,
      count: filtered.length,
      sourceTitle: display.sourcePhoto ? photoTitle(display.sourcePhoto) : state.similarPhotoId,
      pending: Boolean(display.pending),
      activeFilterEntries,
    });
    elements.summary.textContent = text.summary;
    elements.context.textContent = text.context;
  }

  if (photos.length === 0) {
    renderEmpty(elements.grid, "目前資料來源沒有照片資料");
    updateLoadMore({ elements, visibleCount, filtered });
    restoreScroll();
    return;
  }

  if (display.pending && display.mode === "semantic") {
    renderEmpty(elements.grid, "正在計算語意相似度…");
    updateLoadMore({ elements, visibleCount, filtered });
    syncUrlState();
    restoreScroll();
    return;
  }

  if (filtered.length === 0) {
    let emptyText = "沒有命中目前索引的照片";
    if (display.missingSource) {
      emptyText = "這張照片還沒有圖片向量，暫時無法找相似照片";
    } else if (display.mode !== "keyword") {
      emptyText = "符合其他篩選的照片都還沒有向量；可放寬篩選或切換到關鍵字命中";
    }
    renderEmpty(elements.grid, emptyText);
    updateLoadMore({ elements, visibleCount, filtered });
    maybeTrackZeroResults();
    syncUrlState();
    restoreScroll();
    return;
  }

  const fragment = document.createDocumentFragment();
  for (const [index, photo] of filtered.slice(0, visibleCount).entries()) {
    fragment.append(renderPhoto(photo, index + 1, filtered.length));
  }
  elements.grid.append(fragment);
  updateLoadMore({ elements, visibleCount, filtered });
  syncUrlState();

  if (source) {
    scheduleResultsTracking(source);
  }

  restoreScroll();
  queueAutoLoadMore();
}

function maybeTrackZeroResults() {
  const snapshot = currentFilterSnapshot();
  if (!hasTrackedResultState(snapshot)) {
    return;
  }
  const zeroState = JSON.stringify(snapshot);
  if (zeroState === state.lastTrackedZeroState) {
    return;
  }
  state.lastTrackedZeroState = zeroState;
  trackEvent("zero_results", resultsEventParams(snapshot));
}

function resetFilters() {
  state.taskMode = "all";
  state.view = "";
  state.similarPhotoId = "";
  state.similarReturn = null;
  controls.search.value = "";
  controls.sort.value = "recommended";
  for (const definition of filterDefinitions) {
    setFilterValues(definition.key, []);
  }
  syncControlsFromState();
  render({ resetPage: true, source: "filter" });
}

function syncUrlState() {
  const url = new URL(window.location.href);
  url.search = encodeUrlState(currentUrlState()).toString();
  // The remembered pre-similar query rides along in history state so back and
  // forward keep chip removal able to restore it.
  window.history.replaceState(state.similarReturn ? { similarReturn: state.similarReturn } : null, "", url);
}

function applyUrlState() {
  const urlState = decodeUrlState(new URLSearchParams(window.location.search));
  if (taskModes.some((mode) => mode.id === urlState.taskMode)) {
    state.taskMode = urlState.taskMode;
  }
  controls.search.value = urlState.search;
  if (urlState.sort && [...controls.sort.options].some((option) => option.value === urlState.sort)) {
    controls.sort.value = urlState.sort;
  }
  for (const definition of filterDefinitions) {
    setFilterValues(definition.key, urlState.filters[definition.key] ?? [], { syncControl: false });
  }
  for (const photoId of urlState.selectedPhotoIds) {
    state.selectedPhotoIds.add(photoId);
    state.promotedPhotoIds.add(photoId);
  }
  state.view = urlState.view;
  state.similarPhotoId = urlState.similar;
  state.similarReturn = urlState.similar ? window.history.state?.similarReturn ?? null : null;
  syncControlsFromState();
}

async function revealPhotoFromHash() {
  const match = window.location.hash.match(/^#photo-(\d+)$/);
  if (!match) {
    return;
  }

  const targetId = match[1];
  const resultIndex = currentResults.findIndex((photo) => photo.photo_id === targetId);
  const targetPhoto = resultIndex >= 0 ? currentResults[resultIndex] : photos.find((photo) => photo.photo_id === targetId);
  if (!targetPhoto) {
    return;
  }

  if (resultIndex >= visibleCount) {
    visibleCount = Math.ceil((resultIndex + 1) / pageSize) * pageSize;
    render({ preservePage: true });
  }

  const card = document.getElementById(photoAnchorId(targetId));
  if (card) {
    card.scrollIntoView({ block: "center" });
  }

  await openPreview(targetPhoto);
}

async function loadData() {
  const loadedData = await loadFinderData({ dataSources, projectConfigUrl });
  applyControlsRegistry(loadedData.interfaceRegistry);
  applyTaskModeRegistry(loadedData.interfaceRegistry);
  applyUrlStateRegistry(loadedData.interfaceRegistry);
  applySearchRegistry(loadedData.interfaceRegistry);
  refreshFilterControls();
  bindFilterControlEvents();
  for (const definition of filterDefinitions) {
    if (!Object.hasOwn(state.filters, definition.key)) {
      state.filters[definition.key] = [];
    }
  }
  applyProjectConfig(loadedData.projectConfig);
  photoSchema = loadedData.photoSchema;
  optionLabelMaps = loadedData.optionLabelMaps;
  searchTokensForField = loadedData.searchTokensForField;
  albums = loadedData.albums;
  photos = loadedData.photos;
  loadPhotoDetails = loadedData.loadPhotoDetails ?? loadPhotoDetails;
  maybeRegisterPwa(loadedData.finderDataManifest?.generatedAt ?? "");
  setupTaskModes(elements.taskModes, taskModes);
  initializeMobileTaskModePanel();
  setupFilters({
    controls,
    elements,
    taxonomy: loadedData.taxonomy,
    photos,
    albums,
    peopleCountFilters,
    optionLabels,
    uniqueSorted,
  });
  updateFilterLayout({ controls, elements, taskMode: state.taskMode });
  photoRowById = new Map(photos.map((photo, row) => [String(photo.photo_id), row]));
  semantic = createSemanticController({
    metaUrl: dataSources.finderEmbeddingsMetaUrl ?? "",
    photoCount: photos.length,
    photosSha256: loadedData.finderDataManifest?.source?.photosSha256 ?? "",
    trackEvent,
    onChange: onSemanticChange,
  });
  semantic.dialog?.addEventListener("close", () => {
    if (pendingSimilarPhotoId && !["downloading", "loading"].includes(semantic.status)) {
      pendingSimilarPhotoId = "";
    }
  });
  applyUrlState();
  renderOverview();
  render({ resetPage: true });
  revealPhotoFromHash();
  semantic.restore();
}

function scheduleSearchRender() {
  clearTimeout(renderTimer);
  renderTimer = window.setTimeout(() => {
    render({ resetPage: true, source: "search" });
  }, searchDebounceMs);
}

function bindFilterControlEvents() {
  if (filterControlEventsBound) {
    return;
  }
  filterControlEventsBound = true;
  for (const [key, control] of Object.entries(controls)) {
    if ([
      "reset",
      "loadMore",
      "copyCandidates",
      "clearCandidates",
      "candidateCopyMenuButton",
      "candidateCopyMenu",
      "candidateCopyMenuItems",
      "copyAiAssistantPrompt",
      "mobileFilter",
      "mobileCandidate",
      "closeFilterSheet",
      "closeCandidateSheet",
      "closePreview",
      "previewCandidate",
      "previewLarge",
      "previewCopyFlickr",
      "previewCopyFinder",
    ].includes(key)) {
      continue;
    }
    const filterDefinition = filterDefinitions.find((definition) => definition.control === key);
    if (filterDefinition) {
      control.addEventListener("input", () => {
        syncStateFromControl(filterDefinition.key);
        render({ resetPage: true, source: "filter" });
      });
    } else if (key === "search") {
      control.addEventListener("input", () => {
        // Typing a new query replaces a "similar to" condition.
        state.similarPhotoId = "";
        state.similarReturn = null;
        scheduleSearchRender();
      });
    } else {
      control.addEventListener("input", () => render({ resetPage: true, source: "filter" }));
    }
  }
}

elements.taskModes.addEventListener("click", (event) => {
  const button = event.target.closest("[data-task-mode]");
  if (!button) {
    return;
  }
  state.taskMode = button.dataset.taskMode;
  render({ resetPage: true, source: "filter" });
  trackEvent("select_task_mode", { task_mode: state.taskMode });
});

elements.activeFilters.addEventListener("click", (event) => {
  const button = event.target.closest("[data-filter-key]");
  if (!button) {
    return;
  }
  clearFilter(button.dataset.filterKey, button.dataset.filterValue ?? "");
});

controls.reset.addEventListener("click", resetFilters);
controls.loadMore.addEventListener("click", () => {
  loadMoreResults("load_more");
});
controls.copyCandidates.addEventListener("click", copyCandidateLink);
controls.candidateCopyMenuButton.addEventListener("click", toggleCandidateCopyMenu);
controls.candidateCopyMenu.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target : null;
  const button = target?.closest("[data-candidate-copy-template], [data-candidate-batch-action]");
  if (!button) {
    return;
  }
  closeCandidateCopyMenu();
  if (button.dataset.candidateBatchAction) {
    runCandidateBatchAction(button.dataset.candidateBatchAction);
  } else {
    copyCandidateTemplate(button.dataset.candidateCopyTemplate);
  }
});
controls.clearCandidates.addEventListener("click", clearCandidates);
controls.copyAiAssistantPrompt.addEventListener("click", copyAiAssistantPrompt);
controls.mobileFilter.addEventListener("click", openFilterSheet);
controls.mobileCandidate.addEventListener("click", openCandidateSheet);
controls.closeFilterSheet.addEventListener("click", closeFilterSheet);
controls.closeCandidateSheet.addEventListener("click", closeCandidateSheet);
controls.closePreview.addEventListener("click", closePreview);
elements.photoPreviewDialog.addEventListener("cancel", (event) => {
  event.preventDefault();
  closePreview();
});
elements.photoPreviewDialog.addEventListener("click", (event) => {
  if (event.target !== elements.photoPreviewDialog) {
    return;
  }
  const rect = elements.photoPreviewDialog.getBoundingClientRect();
  if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) {
    closePreview();
  }
});
elements.searchPanel.addEventListener(
  "touchstart",
  (event) => onSheetTouchStart(event, {
    sheet: elements.searchPanel,
    isOpen: () => elements.searchPanel.classList.contains("is-filter-open"),
    close: closeFilterSheet,
  }),
  { passive: true },
);
elements.sidePanel.addEventListener(
  "touchstart",
  (event) => onSheetTouchStart(event, {
    sheet: elements.sidePanel,
    isOpen: () => elements.sidePanel.classList.contains("is-candidate-open"),
    close: closeCandidateSheet,
  }),
  { passive: true },
);
elements.photoPreviewDialog.addEventListener(
  "touchstart",
  (event) => onSheetTouchStart(event, {
    sheet: elements.photoPreviewDialog,
    isOpen: () => !elements.photoPreviewDialog.hidden,
    close: closePreview,
  }),
  { passive: true },
);
elements.previewImageFrame.addEventListener(
  "touchstart",
  (event) => {
    if (!isPreviewOpen() || !isMobileSheet() || event.touches.length !== 1) {
      previewSwipeState = null;
      return;
    }
    const touch = event.touches[0];
    previewSwipeState = {
      touchId: touch.identifier,
      startX: touch.clientX,
      startY: touch.clientY,
    };
  },
  { passive: true },
);
elements.previewImageFrame.addEventListener(
  "touchmove",
  (event) => {
    const touch = previewSwipeState ? touchFromList(event.touches, previewSwipeState.touchId) : null;
    if (!touch) {
      return;
    }
    const deltaX = touch.clientX - previewSwipeState.startX;
    const deltaY = touch.clientY - previewSwipeState.startY;
    if (Math.abs(deltaX) > 12 && Math.abs(deltaX) > Math.abs(deltaY)) {
      event.preventDefault();
    }
  },
  { passive: false },
);
elements.previewImageFrame.addEventListener("touchend", (event) => {
  const touch = previewSwipeState ? touchFromList(event.changedTouches, previewSwipeState.touchId) : null;
  if (!touch) {
    previewSwipeState = null;
    return;
  }
  const step = horizontalSwipeStep({
    startX: previewSwipeState.startX,
    startY: previewSwipeState.startY,
    endX: touch.clientX,
    endY: touch.clientY,
  });
  previewSwipeState = null;
  if (step && navigatePreview(step)) {
    suppressPreviewImageClick = true;
    window.setTimeout(() => {
      suppressPreviewImageClick = false;
    }, 0);
    event.preventDefault();
  }
});
elements.previewImageFrame.addEventListener("touchcancel", () => {
  previewSwipeState = null;
});
elements.previewImageLink.addEventListener("click", (event) => {
  if (!suppressPreviewImageClick) {
    return;
  }
  suppressPreviewImageClick = false;
  event.preventDefault();
});
for (const sheet of [elements.searchPanel, elements.sidePanel, elements.photoPreviewDialog]) {
  sheet.addEventListener("touchmove", onSheetTouchMove, { passive: false });
  sheet.addEventListener("touchend", onSheetTouchEnd);
  sheet.addEventListener("touchcancel", onSheetTouchEnd);
}
elements.resultTabs.addEventListener("click", (event) => {
  const tab = event.target instanceof Element ? event.target.closest("[data-result-view]") : null;
  if (!tab || tab.dataset.resultView === currentView.mode) {
    return;
  }
  state.view = tab.dataset.resultView;
  render({ resetPage: true, source: "search" });
  tab.focus({ preventScroll: true });
});
elements.resultTabs.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") {
    return;
  }
  event.preventDefault();
  event.stopPropagation();
  const next = currentView.mode === "semantic" ? elements.keywordTab : elements.semanticTab;
  next.click();
});
elements.previewSimilar.addEventListener("click", findSimilarToPreview);
controls.previewLarge.addEventListener("click", downloadPreviewLargeImage);
controls.previewCopyFlickr.addEventListener("click", copyPreviewFlickrLink);
controls.previewCopyFinder.addEventListener("click", copyPreviewFinderLink);
document.addEventListener("click", (event) => {
  const target = event.target instanceof Element ? event.target : null;
  if (controls.candidateCopyMenu.hidden || target?.closest(".candidate-copy-menu")) {
    return;
  }
  closeCandidateCopyMenu();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeCandidateCopyMenu();
  }
});
controls.previewCandidate.addEventListener("click", () => {
  if (activePreviewPhoto) {
    toggleCandidate(activePreviewPhoto.photo_id);
  }
});
elements.modalBackdrop.addEventListener("click", closeMobileOverlays);
elements.aiAssistantSheetLink.addEventListener("click", () => {
  trackEvent("open_sheets_for_ai_assistant", currentAiAssistantEventParams());
});
window.addEventListener("hashchange", revealPhotoFromHash);
window.addEventListener("popstate", () => {
  const urlState = decodeUrlState(new URLSearchParams(window.location.search));
  if (
    urlState.search === controls.search.value &&
    urlState.view === state.view &&
    urlState.similar === state.similarPhotoId
  ) {
    return;
  }
  const similarReturn = urlState.similar ? window.history.state?.similarReturn ?? null : null;
  applySimilarState({ search: urlState.search, view: urlState.view, similar: urlState.similar, similarReturn });
  render({ resetPage: true, source: "search" });
});
window.addEventListener("scroll", queueAutoLoadMore, { passive: true });
window.addEventListener("resize", queueAutoLoadMore);
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeMobileOverlays();
    return;
  }
  if (!isMobileSheet() && event.key === "ArrowLeft" && navigatePreview(-1)) {
    event.preventDefault();
    return;
  }
  if (!isMobileSheet() && event.key === "ArrowRight" && navigatePreview(1)) {
    event.preventDefault();
  }
});

try {
  await loadData();
} catch (error) {
  elements.grid.setAttribute("aria-busy", "false");
  elements.summary.textContent = "資料載入失敗";
  elements.context.textContent = "請確認資料來源與公開讀取權限";
  elements.overviewSummary.textContent = "資料載入失敗";
  elements.overviewGrid.replaceChildren();
  elements.grid.replaceChildren();
  renderEmpty(elements.grid, error.message);
}
