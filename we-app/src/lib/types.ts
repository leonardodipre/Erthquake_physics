export interface MetricsSummary {
  sample_count?: number;
  step_count?: number;
  final_total_loss?: number | null;
  best_total_loss?: number | null;
  final_slip_max_m?: number | null;
  final_a_minus_b_mean?: number | null;
  median_rmse_mm?: number | null;
  p90_rmse_mm?: number | null;
  worst_rmse_mm?: number | null;
  median_velocity_rmse_mm_per_day?: number | null;
  date_range?: {
    start: string | null;
    end: string | null;
  };
}

export interface ManifestData {
  app_title: string;
  generated_at: string;
  paths: {
    fault_geojson: string;
    fault_patches: string;
    station_metadata: string;
    stations: string;
    timeseries_dir: string;
    models_index: string;
    snapshots_index: string;
    metrics: {
      evaluation: string;
      training_history: string;
      station_sensitivity: string;
    };
  };
  summary: {
    fault_patches: number;
    stations: number;
    snapshots: number;
    models?: number;
    coordinate_source: string;
  };
  metrics_summary: {
    evaluation: MetricsSummary;
    training_history: MetricsSummary;
  };
}

export interface StationEntry {
  station_id: string;
  label: string;
  category: "accepted" | "modified" | "acc_test" | string;
  reference_system: string;
  latitude: number;
  longitude: number;
  utm_e_m: number;
  utm_n_m: number;
  local_xyz_m?: [number, number, number];
  coordinates_source: string;
  is_fallback_coordinate: boolean;
  networks?: string;
  distance_km_from_laquila?: number | null;
  available_date_range: {
    start: string;
    end: string;
  };
  timeseries_path: string;
  sample_count: number;
  source_file: string;
}

export interface SeismicEventReference {
  name: string;
  origin_time_utc: string;
  magnitude_mw: number;
  latitude: number;
  longitude: number;
  depth_m: number;
  utm_e_m: number;
  utm_n_m: number;
  local_xyz_m: [number, number, number];
  source: string;
}

export interface StationsData {
  meta: {
    station_count: number;
    category_counts: Record<string, number>;
    coordinate_source: string;
    real_coordinate_count?: number;
    fallback_coordinate_count?: number;
    total_time_series_samples: number;
    reference_point?: {
      name: string;
      latitude: number;
      longitude: number;
      utm_e_m: number;
      utm_n_m: number;
      local_xyz_m: [number, number, number];
      max_distance_km?: number | null;
    } | null;
    mainshock?: SeismicEventReference | null;
  };
  stations: StationEntry[];
}

export interface StationTimeseries {
  station_id: string;
  label: string;
  category: string;
  source_file: string;
  reference_system: string;
  units: {
    displacement: string;
  };
  date_range: {
    start: string;
    end: string;
  };
  has_cleaned: boolean;
  dates: string[];
  raw: {
    E: Array<number | null>;
    N: Array<number | null>;
    U: Array<number | null>;
  };
  cleaned: {
    E: Array<number | null>;
    N: Array<number | null>;
    U: Array<number | null>;
  };
}

export interface FaultPatch {
  id: number;
  row: number;
  col: number;
  center_enu_m: [number, number, number];
  center_local_xyz_m: [number, number, number];
  center_lon_lat: [number, number];
  depth_m: number;
  corners_enu_m: [number, number, number][];
  corners_local_xyz_m: [number, number, number][];
  triangles_enu_m: [number, number, number][][];
  triangles_local_xyz_m: [number, number, number][][];
}

export interface FaultPatchesData {
  meta: {
    patch_count: number;
    triangle_count: number;
    grid: {
      nx: number;
      ny: number;
    };
    utm_epsg: number;
    local_origin_utm_en_m: [number, number];
    fault_meta: {
      strike_deg: number;
      dip_deg: number;
      rake_deg: number;
      top_depth_m: number;
      bottom_depth_m: number;
      length_m: number;
      width_m: number;
      origin_en: [number, number];
    };
    elastic: {
      mu: number;
      nu: number;
    };
    notes: Record<string, string>;
  };
  surface_trace_local_xyz_m?: [number, number, number][];
  patches: FaultPatch[];
}

export interface SnapshotFieldMeta {
  label: string;
  units: string;
  color_map: string;
  scale: "linear" | "log" | "symmetric";
  min: number;
  max: number;
}

export interface SnapshotIndex {
  meta: {
    model_key?: string;
    model_label?: string;
    checkpoint: string;
    evaluation_csv?: string;
    history_csv?: string | null;
    reference_date: string;
    snapshot_count: number;
    baseline_snapshot_date: string | null;
    patch_count: number;
    station_count: number;
  };
  fields: Record<string, SnapshotFieldMeta>;
  static_fields?: Record<string, number[]>;
  snapshots: Array<{
    date: string;
    date_key: string;
    path: string;
  }>;
}

export interface SnapshotData {
  date: string;
  time_seconds: number;
  fields: Record<string, number[]>;
  surface_predictions: {
    station_ids: string[];
    u_surface_m: [number, number, number][];
    v_surface_m_per_s: [number, number, number][];
  };
}

export interface EvaluationMetricsData {
  summary: MetricsSummary;
  records: Array<Record<string, string | number>>;
}

export interface TrainingHistoryData {
  summary: MetricsSummary;
  records: Array<Record<string, string | number>>;
}

export interface ModelCatalogEntry {
  key: string;
  label: string;
  checkpoint: string;
  evaluation_csv: string;
  history_csv: string | null;
  snapshot_index_path: string;
  station_sensitivity_path: string | null;
  snapshot_count: number;
  time_range: {
    start: string | null;
    end: string | null;
  };
  metrics_summary: MetricsSummary;
  history_summary: MetricsSummary | null;
  station_sensitivity_summary?: {
    station_count: number;
    max_sensitivity: number;
    median_sensitivity: number;
    top_station: string;
  } | null;
}

export interface ModelCatalog {
  default_model_key: string;
  models: ModelCatalogEntry[];
}

export interface StationSensitivityRecord {
  rank: number;
  station: string;
  sensitivity: number;
}
