import type {
  EvaluationMetricsData,
  FaultPatchesData,
  ManifestData,
  ModelCatalog,
  SnapshotData,
  SnapshotIndex,
  StationSensitivityRecord,
  StationsData,
  StationTimeseries,
  TrainingHistoryData,
} from "./types";

const jsonCache = new Map<string, Promise<unknown>>();

async function fetchJson<T>(path: string): Promise<T> {
  const cached = jsonCache.get(path);
  if (cached) {
    return cached as Promise<T>;
  }

  const pending = fetch(path).then(async (response) => {
    if (!response.ok) {
      throw new Error(`Failed to fetch ${path}: ${response.status}`);
    }
    return (await response.json()) as T;
  });
  jsonCache.set(path, pending);
  return pending;
}

export function loadManifest() {
  return fetchJson<ManifestData>("/data/manifest.json");
}

export function loadStations() {
  return fetchJson<StationsData>("/data/stations.json");
}

export function loadStationTimeseries(stationId: string) {
  return fetchJson<StationTimeseries>(`/data/timeseries/${stationId}.json`);
}

export function loadFaultGeoJSON() {
  return fetchJson<GeoJSON.FeatureCollection>("/data/fault.geojson");
}

export function loadFaultPatches() {
  return fetchJson<FaultPatchesData>("/data/fault_patches.json");
}

export function loadSnapshotIndex() {
  return fetchJson<SnapshotIndex>("/data/model_snapshots/index.json");
}

export function loadModelCatalog() {
  return fetchJson<ModelCatalog>("/data/models/index.json");
}

export function loadSnapshotIndexForModel(modelKey: string) {
  return fetchJson<SnapshotIndex>(`/data/model_snapshots/${modelKey}/index.json`);
}

export function loadSnapshot(dateKey: string, modelKey?: string) {
  const prefix = modelKey ? `/data/model_snapshots/${modelKey}` : "/data/model_snapshots";
  return fetchJson<SnapshotData>(`${prefix}/${dateKey}.json`);
}

export function loadStationSensitivity(path: string) {
  return fetchJson<StationSensitivityRecord[]>(path);
}

export function loadEvaluationMetrics() {
  return fetchJson<EvaluationMetricsData>("/data/metrics/evaluation.json");
}

export function loadTrainingHistory() {
  return fetchJson<TrainingHistoryData>("/data/metrics/training_history.json");
}
