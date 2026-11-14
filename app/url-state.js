// URL query serialization for the Pages finder. Keep this independent from DOM
// controls so deep-link behavior stays reviewable and testable.
let filterUrlKeys = {
  album: "album",
  use: "use",
  mood: "mood",
  scene: "scene",
  peopleCount: "people",
  subjectType: "subject",
  orientation: "orientation",
  negativeSpace: "negative",
  safeCrop: "crop",
  sponsorshipTag: "sponsorTag",
  sponsorshipItem: "sponsorItem",
  publicStatus: "public",
  priority: "priority",
  curationStatus: "curation",
  collection: "collection",
};

export function applyUrlStateRegistry(interfaceRegistry) {
  const filters = interfaceRegistry?.pages?.filters ?? [];
  if (filters.length > 0) {
    filterUrlKeys = Object.fromEntries(filters.map((filter) => [filter.key, filter.urlKey]));
  }
}

function cleanValues(values) {
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

function appendValues(params, key, values) {
  for (const value of cleanValues(values)) {
    params.append(key, value);
  }
}

export function encodeUrlState(state) {
  const params = new URLSearchParams();

  if (state.taskMode && state.taskMode !== "all") {
    params.set("task", state.taskMode);
  }
  const search = String(state.search ?? "").trim();
  if (search) {
    params.set("q", search);
  }
  if (state.sort && state.sort !== "recommended") {
    params.set("sort", state.sort);
  }

  for (const [filterKey, urlKey] of Object.entries(filterUrlKeys)) {
    appendValues(params, urlKey, state.filters?.[filterKey] ?? []);
  }

  const selected = [...(state.selectedPhotoIds ?? [])].map((value) => String(value ?? "").trim()).filter(Boolean);
  if (selected.length > 0) {
    params.set("selected", selected.join(","));
  }
  // Semantic views only describe what was shown; whether semantic search is
  // enabled is a per-device setting and never goes into the URL.
  const view = cleanResultView(state.view);
  if (view) {
    params.set("view", view);
  }
  const similar = String(state.similar ?? "").trim();
  if (similar) {
    params.set("similar", similar);
  }
  return params;
}

export const resultViews = ["semantic", "keyword"];

function cleanResultView(value) {
  const view = String(value ?? "").trim();
  return resultViews.includes(view) ? view : "";
}

export function finderStateUrl(href, state) {
  const url = new URL(href);
  url.hash = "";
  url.search = encodeUrlState(state).toString();
  return url.toString();
}

export function decodeUrlState(params) {
  const filters = {};
  for (const [filterKey, urlKey] of Object.entries(filterUrlKeys)) {
    filters[filterKey] = cleanValues(params.getAll(urlKey));
  }

  return {
    taskMode: params.get("task") ?? "",
    search: params.get("q") ?? "",
    sort: params.get("sort") ?? "",
    filters,
    selectedPhotoIds: (params.get("selected") ?? "").split(",").filter(Boolean),
    view: cleanResultView(params.get("view")),
    similar: String(params.get("similar") ?? "").trim(),
  };
}
