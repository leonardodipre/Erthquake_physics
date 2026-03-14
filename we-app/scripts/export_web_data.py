from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from pyproj import Transformer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import DEFAULT_DATA_SUBDIRS, extract_marker_from_filename  # noqa: E402
from predict import load_trained_model  # noqa: E402


SECONDS_PER_DAY = 86_400.0

LAQUILA_MAINSHOCK = {
    "name": "2009 L'Aquila mainshock",
    "origin_time_utc": "2009-04-06T01:32:40Z",
    "magnitude_mw": 6.3,
    "latitude": 42.3420,
    "longitude": 13.3800,
    "depth_m": 8_000.0,
    "source": "https://eida.ingv.it/en/network/8H_2009",
}


@dataclass(frozen=True)
class ExportPaths:
    public_dir: Path
    data_dir: Path
    timeseries_dir: Path
    snapshots_dir: Path
    metrics_dir: Path


@dataclass(frozen=True)
class ModelExportSpec:
    key: str
    label: str
    checkpoint_path: Path
    eval_csv_path: Path
    history_csv_path: Path | None
    station_sensitivity_path: Path | None


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export browser-friendly scientific assets")
    parser.add_argument("--checkpoint", default="checkpoints/8_V5/8_V5_alldata_bigmodel.pt")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--data-dir", default="dataset_scremato")
    parser.add_argument(
        "--stations-json",
        default="dataset_scremato/stations_ITA_laquila_150km.json",
        help="Local station metadata JSON with marker/aprioriNorth/aprioriEast fields.",
    )
    parser.add_argument("--fault-geojson", default="dataset_scremato/faglia_aquila.geojson")
    parser.add_argument("--eval-csv", default="checkpoints/8_V5/8_V5.csv")
    parser.add_argument(
        "--history-csv",
        default="checkpoints/8_V5/8_V5_alldata_bigmodel.history.csv",
    )
    parser.add_argument(
        "--station-sensitivity-json",
        default="checkpoints/8_V5/diagnostics/08_station_sensitivity_ranking.json",
    )
    parser.add_argument("--output-dir", default="we-app/public/data")
    parser.add_argument(
        "--snapshot-frequency",
        default="MS",
        help="Pandas frequency for snapshot exports. Default: monthly starts (MS).",
    )
    return parser.parse_args()


def _ensure_export_dirs(output_dir: Path) -> ExportPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeseries_dir = output_dir / "timeseries"
    snapshots_dir = output_dir / "model_snapshots"
    metrics_dir = output_dir / "metrics"
    for directory in (timeseries_dir, snapshots_dir, metrics_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return ExportPaths(
        public_dir=output_dir.parent,
        data_dir=output_dir,
        timeseries_dir=timeseries_dir,
        snapshots_dir=snapshots_dir,
        metrics_dir=metrics_dir,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=False)
    else:
        text = json.dumps(payload, separators=(",", ":"), sort_keys=False)
    path.write_text(text + "\n", encoding="utf-8")


def _read_fault_geojson(path: Path) -> dict[str, Any]:
    data = _load_json(path)
    if data.get("type") != "FeatureCollection":
        raise ValueError(f"Unexpected fault GeoJSON type in {path}: {data.get('type')!r}")
    return data


def _patch_basis(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = corners.mean(axis=0)
    basis_u = corners[0] - centroid
    basis_u = basis_u / max(np.linalg.norm(basis_u), 1e-12)
    normal = np.cross(corners[1] - corners[0], corners[2] - corners[0])
    normal = normal / max(np.linalg.norm(normal), 1e-12)
    basis_v = np.cross(normal, basis_u)
    basis_v = basis_v / max(np.linalg.norm(basis_v), 1e-12)
    return centroid, basis_u, basis_v


def _ordered_corners(vertices: np.ndarray) -> np.ndarray:
    rounded = np.round(vertices, decimals=6)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    corners = vertices[np.sort(unique_idx)]
    if corners.shape[0] != 4:
        raise ValueError(f"Expected 4 unique patch corners, found {corners.shape[0]}")

    centroid, basis_u, basis_v = _patch_basis(corners)
    rel = corners - centroid
    proj_u = rel @ basis_u
    proj_v = rel @ basis_v
    angles = np.arctan2(proj_v, proj_u)
    order = np.argsort(angles)
    return corners[order]


def _to_local_xyz(points_enu: np.ndarray, origin_e: float, origin_n: float) -> np.ndarray:
    points = np.asarray(points_enu, dtype=np.float64)
    local = np.empty_like(points)
    local[:, 0] = points[:, 0] - origin_e
    local[:, 1] = points[:, 2]
    local[:, 2] = points[:, 1] - origin_n
    return local


def export_fault_geometry(
    *,
    fault_geojson_path: Path,
    green_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    fault_geojson = _read_fault_geojson(fault_geojson_path)
    target_geojson = output_dir / "fault.geojson"
    shutil.copy2(fault_geojson_path, target_geojson)

    summary = _load_json(green_dir / "green_summary.json")
    mesh = np.load(green_dir / "fault_mesh.npz", allow_pickle=True)

    tris = np.asarray(mesh["tris"], dtype=np.float64)
    centers = np.asarray(mesh["patch_centers"], dtype=np.float64)
    utm_epsg = int(summary["utm_epsg"])
    nx = int(summary["nx"])
    ny = int(summary["ny"])
    origin_e, origin_n = centers[:, 0].mean(), centers[:, 1].mean()

    if tris.shape[0] != centers.shape[0] * 2:
        raise ValueError(
            f"Expected two triangles per patch, found {tris.shape[0]} triangles for {centers.shape[0]} patches."
        )

    transformer = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    patch_entries: list[dict[str, Any]] = []

    for patch_id in range(centers.shape[0]):
        patch_tris = tris[2 * patch_id : 2 * patch_id + 2]
        all_vertices = patch_tris.reshape(-1, 3)
        corners = _ordered_corners(all_vertices)
        center = centers[patch_id]
        lon, lat = transformer.transform(center[0], center[1])

        patch_entries.append(
            {
                "id": patch_id,
                "row": patch_id // nx,
                "col": patch_id % nx,
                "center_enu_m": center.tolist(),
                "center_local_xyz_m": _to_local_xyz(center.reshape(1, 3), origin_e, origin_n)[0].tolist(),
                "center_lon_lat": [lon, lat],
                "depth_m": float(-center[2]),
                "corners_enu_m": corners.tolist(),
                "corners_local_xyz_m": _to_local_xyz(corners, origin_e, origin_n).tolist(),
                "triangles_enu_m": patch_tris.tolist(),
                "triangles_local_xyz_m": _to_local_xyz(
                    patch_tris.reshape(-1, 3), origin_e, origin_n
                ).reshape(2, 3, 3).tolist(),
            }
        )

    trace_coords = fault_geojson["features"][0]["geometry"]["coordinates"][0]
    trace_lon = np.asarray([point[0] for point in trace_coords], dtype=np.float64)
    trace_lat = np.asarray([point[1] for point in trace_coords], dtype=np.float64)
    trace_e, trace_n = to_utm.transform(trace_lon, trace_lat)
    trace_enu = np.column_stack(
        [np.asarray(trace_e, dtype=np.float64), np.asarray(trace_n, dtype=np.float64), np.zeros_like(trace_e)]
    )

    payload = {
        "meta": {
            "patch_count": int(centers.shape[0]),
            "triangle_count": int(tris.shape[0]),
            "grid": {"nx": nx, "ny": ny},
            "utm_epsg": utm_epsg,
            "local_origin_utm_en_m": [float(origin_e), float(origin_n)],
            "fault_meta": summary["fault_meta"],
            "elastic": summary.get("elastic", {}),
            "notes": summary.get("notes", {}),
        },
        "surface_trace_local_xyz_m": _to_local_xyz(trace_enu, origin_e, origin_n).tolist(),
        "patches": patch_entries,
    }
    _write_json(output_dir / "fault_patches.json", payload, pretty=False)
    return payload


def _station_file_index(
    data_dir: Path,
    station_ids: Iterable[str],
) -> dict[str, dict[str, Any]]:
    station_set = {str(station_id).strip().upper() for station_id in station_ids}
    selected: dict[str, dict[str, Any]] = {}

    for subdir in DEFAULT_DATA_SUBDIRS:
        folder = data_dir / subdir
        if not folder.exists():
            continue
        for csv_path in sorted(folder.glob("*.csv")):
            marker = extract_marker_from_filename(csv_path)
            if marker in station_set and marker not in selected:
                selected[marker] = {
                    "path": csv_path,
                    "category": subdir,
                }

    missing = sorted(station_set - set(selected))
    if missing:
        raise FileNotFoundError(
            f"Missing time series files for {len(missing)} model stations, e.g. {missing[:5]}"
        )

    return selected


def _load_station_positions_from_json(
    stations_json_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], str, dict[str, Any] | None]:
    if stations_json_path is None or not stations_json_path.exists():
        return {}, "mock_generated_from_fault_extent", None

    payload = _load_json(stations_json_path)
    stations_raw = payload.get("stations", payload) if isinstance(payload, dict) else payload
    if not isinstance(stations_raw, list):
        raise ValueError(f"Unsupported stations JSON structure in {stations_json_path}")

    positions: dict[str, dict[str, Any]] = {}
    for station in stations_raw:
        marker = str(station.get("marker", "")).strip().upper()
        if not marker:
            continue
        if "aprioriNorth" not in station or "aprioriEast" not in station:
            continue
        positions[marker] = {
            "station_id": marker,
            "latitude": float(station["aprioriNorth"]),
            "longitude": float(station["aprioriEast"]),
            "utm_e_m": None,
            "utm_n_m": None,
            "apriori_up_m": float(station.get("aprioriUp", 0.0)),
            "networks": str(station.get("networks", "")).strip(),
            "distance_km_from_laquila": (
                float(station["distance_km_from_laquila"])
                if station.get("distance_km_from_laquila") is not None
                else None
            ),
            "source_station_id": station.get("id"),
        }

    reference = None
    if isinstance(payload, dict) and isinstance(payload.get("reference"), dict):
        ref = payload["reference"]
        if ref.get("lat") is not None and ref.get("lon") is not None:
            reference = {
                "name": str(ref.get("name", "Reference")).strip() or "Reference",
                "latitude": float(ref["lat"]),
                "longitude": float(ref["lon"]),
                "max_distance_km": (
                    float(ref["max_distance_km"]) if ref.get("max_distance_km") is not None else None
                ),
            }

    return positions, str(stations_json_path.relative_to(REPO_ROOT)), reference


def _sanitize_model_key(raw_value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in raw_value)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_").lower()


def _discover_model_specs(
    checkpoints_root: Path,
    default_checkpoint_path: Path,
    default_eval_csv_path: Path,
    default_history_csv_path: Path,
) -> list[ModelExportSpec]:
    specs_by_key: dict[str, ModelExportSpec] = {}

    for checkpoint_path in sorted(checkpoints_root.glob("*/*.pt")):
        if "diagnostics" in checkpoint_path.parts[-2].lower():
            continue
        parent = checkpoint_path.parent
        eval_candidates = sorted(
            path for path in parent.glob("*.csv") if "history" not in path.name.lower()
        )
        if not eval_candidates:
            continue
        history_candidates = sorted(
            path for path in parent.glob("*.csv") if "history" in path.name.lower()
        )
        label = parent.name
        key = _sanitize_model_key(label)
        specs_by_key[key] = ModelExportSpec(
            key=key,
            label=label,
            checkpoint_path=checkpoint_path,
            eval_csv_path=eval_candidates[0],
            history_csv_path=history_candidates[0] if history_candidates else None,
            station_sensitivity_path=(
                parent / "diagnostics" / "08_station_sensitivity_ranking.json"
                if (parent / "diagnostics" / "08_station_sensitivity_ranking.json").exists()
                else None
            ),
        )

    default_key = _sanitize_model_key(default_checkpoint_path.parent.name)
    specs_by_key[default_key] = ModelExportSpec(
        key=default_key,
        label=default_checkpoint_path.parent.name,
        checkpoint_path=default_checkpoint_path,
        eval_csv_path=default_eval_csv_path,
        history_csv_path=default_history_csv_path if default_history_csv_path.exists() else None,
        station_sensitivity_path=(
            default_checkpoint_path.parent / "diagnostics" / "08_station_sensitivity_ranking.json"
            if (default_checkpoint_path.parent / "diagnostics" / "08_station_sensitivity_ranking.json").exists()
            else None
        ),
    )

    return [specs_by_key[key] for key in sorted(specs_by_key)]


def _stable_station_seed(station_id: str) -> int:
    digest = hashlib.sha1(station_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _mock_station_positions(
    station_ids: list[str],
    *,
    utm_epsg: int,
    fault_meta: dict[str, Any],
    origin_e: float,
    origin_n: float,
) -> list[dict[str, float]]:
    forward = Transformer.from_crs(f"EPSG:{utm_epsg}", "EPSG:4326", always_xy=True)

    strike_deg = float(fault_meta["strike_deg"])
    strike_rad = math.radians(strike_deg)
    strike_vec = np.array([math.sin(strike_rad), math.cos(strike_rad)], dtype=np.float64)
    cross_vec = np.array([math.sin(strike_rad + math.pi / 2.0), math.cos(strike_rad + math.pi / 2.0)])

    length_m = float(fault_meta["length_m"])
    semi_major = max(length_m * 0.65, 40_000.0)
    semi_minor = 38_000.0
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))

    positions: list[dict[str, float]] = []
    total = max(len(station_ids), 1)
    for index, station_id in enumerate(station_ids):
        radial = math.sqrt((index + 0.5) / total)
        angle = index * golden_angle
        seed = _stable_station_seed(station_id)
        jitter = np.random.default_rng(seed)
        along = math.cos(angle) * semi_major * radial + jitter.normal(0.0, 2_000.0)
        cross = math.sin(angle) * semi_minor * radial + jitter.normal(0.0, 2_500.0)

        east = origin_e + along * strike_vec[0] + cross * cross_vec[0]
        north = origin_n + along * strike_vec[1] + cross * cross_vec[1]
        lon, lat = forward.transform(east, north)
        positions.append(
            {
                "station_id": station_id,
                "longitude": float(lon),
                "latitude": float(lat),
                "utm_e_m": float(east),
                "utm_n_m": float(north),
            }
        )

    return positions


def _sanitize_series(values: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for value in values.to_numpy():
        if pd.isna(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def export_stations(
    *,
    data_dir: Path,
    output_dir: Path,
    station_ids_path: Path,
    fault_payload: dict[str, Any],
    stations_json_path: Path | None,
) -> dict[str, Any]:
    station_ids = [str(item).strip().upper() for item in np.load(station_ids_path, allow_pickle=True).tolist()]
    station_files = _station_file_index(data_dir, station_ids)
    fault_meta = fault_payload["meta"]["fault_meta"]
    utm_epsg = int(fault_payload["meta"]["utm_epsg"])
    origin_e, origin_n = fault_payload["meta"]["local_origin_utm_en_m"]
    forward_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_epsg}", always_xy=True)
    real_positions, position_source, reference_point = _load_station_positions_from_json(stations_json_path)

    fallback_positions = {
        item["station_id"]: item
        for item in _mock_station_positions(
            station_ids,
            utm_epsg=utm_epsg,
            fault_meta=fault_meta,
            origin_e=float(origin_e),
            origin_n=float(origin_n),
        )
    }

    metadata_positions: dict[str, dict[str, Any]] = {}
    for station_id in station_ids:
        real_entry = real_positions.get(station_id)
        if real_entry is not None:
            utm_e, utm_n = forward_utm.transform(real_entry["longitude"], real_entry["latitude"])
            metadata_positions[station_id] = {
                **real_entry,
                "utm_e_m": float(utm_e),
                "utm_n_m": float(utm_n),
                "is_fallback": False,
            }
        else:
            metadata_positions[station_id] = {
                **fallback_positions[station_id],
                "apriori_up_m": 0.0,
                "networks": "",
                "distance_km_from_laquila": None,
                "source_station_id": None,
                "is_fallback": True,
            }

    n_real = sum(1 for item in metadata_positions.values() if not item["is_fallback"])
    n_fallback = len(metadata_positions) - n_real
    coordinate_source = position_source if n_real > 0 else "mock_generated_from_fault_extent"
    reference_payload = None
    if reference_point is not None:
        ref_e, ref_n = forward_utm.transform(reference_point["longitude"], reference_point["latitude"])
        ref_local = _to_local_xyz(np.array([[ref_e, ref_n, 0.0]], dtype=np.float64), float(origin_e), float(origin_n))[0]
        reference_payload = {
            **reference_point,
            "utm_e_m": float(ref_e),
            "utm_n_m": float(ref_n),
            "local_xyz_m": ref_local.tolist(),
        }

    mainshock_e, mainshock_n = forward_utm.transform(
        LAQUILA_MAINSHOCK["longitude"],
        LAQUILA_MAINSHOCK["latitude"],
    )
    mainshock_local = _to_local_xyz(
        np.array(
            [[mainshock_e, mainshock_n, -float(LAQUILA_MAINSHOCK["depth_m"])]],
            dtype=np.float64,
        ),
        float(origin_e),
        float(origin_n),
    )[0]
    mainshock_payload = {
        **LAQUILA_MAINSHOCK,
        "utm_e_m": float(mainshock_e),
        "utm_n_m": float(mainshock_n),
        "local_xyz_m": mainshock_local.tolist(),
    }

    station_metadata_payload = {
        "coordinate_source": coordinate_source,
        "notes": (
            "Real station coordinates loaded from local JSON when available; fallback coordinates "
            "are only used for markers missing from the metadata source."
        ),
        "crs": {
            "geographic": "EPSG:4326",
            "projected": f"EPSG:{utm_epsg}",
        },
        "counts": {
            "total": len(station_ids),
            "real_coordinates": n_real,
            "fallback_coordinates": n_fallback,
        },
        "reference_point": reference_payload,
        "mainshock": mainshock_payload,
        "stations": [],
    }

    station_entries: list[dict[str, Any]] = []
    total_samples = 0
    category_counts: dict[str, int] = {}

    for station_id in station_ids:
        file_info = station_files[station_id]
        csv_path = file_info["path"]
        category = str(file_info["category"])
        category_counts[category] = category_counts.get(category, 0) + 1

        frame = pd.read_csv(csv_path)
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values("date", kind="stable").reset_index(drop=True)

        reference_system = str(frame["referenceSystem"].iloc[0]) if "referenceSystem" in frame.columns else ""
        ts_payload = {
            "station_id": station_id,
            "label": station_id,
            "category": category,
            "source_file": str(csv_path.relative_to(REPO_ROOT)),
            "reference_system": reference_system,
            "units": {"displacement": "m"},
            "date_range": {
                "start": frame["date"].iloc[0].isoformat(),
                "end": frame["date"].iloc[-1].isoformat(),
            },
            "has_cleaned": all(column in frame.columns for column in ("E_clean", "N_clean", "U_clean")),
            "dates": [timestamp.isoformat() for timestamp in frame["date"]],
            "raw": {
                "E": _sanitize_series(frame["E"]),
                "N": _sanitize_series(frame["N"]),
                "U": _sanitize_series(frame["U"]),
            },
            "cleaned": {
                "E": _sanitize_series(frame["E_clean"]) if "E_clean" in frame.columns else [],
                "N": _sanitize_series(frame["N_clean"]) if "N_clean" in frame.columns else [],
                "U": _sanitize_series(frame["U_clean"]) if "U_clean" in frame.columns else [],
            },
        }

        _write_json(output_dir / "timeseries" / f"{station_id}.json", ts_payload, pretty=False)
        total_samples += len(ts_payload["dates"])

        metadata_entry = {
            "station_id": station_id,
            "label": station_id,
            **metadata_positions[station_id],
        }
        station_metadata_payload["stations"].append(metadata_entry)

        station_entries.append(
            {
                "station_id": station_id,
                "label": station_id,
                "category": category,
                "reference_system": reference_system,
                "latitude": metadata_entry["latitude"],
                "longitude": metadata_entry["longitude"],
                "utm_e_m": metadata_entry["utm_e_m"],
                "utm_n_m": metadata_entry["utm_n_m"],
                "local_xyz_m": _to_local_xyz(
                    np.array([[metadata_entry["utm_e_m"], metadata_entry["utm_n_m"], 0.0]], dtype=np.float64),
                    float(origin_e),
                    float(origin_n),
                )[0].tolist(),
                "coordinates_source": coordinate_source if not metadata_entry["is_fallback"] else "mock_generated_from_fault_extent",
                "is_fallback_coordinate": bool(metadata_entry["is_fallback"]),
                "networks": metadata_entry["networks"],
                "distance_km_from_laquila": metadata_entry["distance_km_from_laquila"],
                "available_date_range": ts_payload["date_range"],
                "timeseries_path": f"/data/timeseries/{station_id}.json",
                "sample_count": len(ts_payload["dates"]),
                "source_file": str(csv_path.relative_to(REPO_ROOT)),
            }
        )

    stations_payload = {
        "meta": {
            "station_count": len(station_entries),
            "category_counts": category_counts,
            "coordinate_source": coordinate_source,
            "real_coordinate_count": n_real,
            "fallback_coordinate_count": n_fallback,
            "total_time_series_samples": total_samples,
            "reference_point": reference_payload,
            "mainshock": mainshock_payload,
        },
        "stations": station_entries,
    }

    _write_json(output_dir / "station_metadata.json", station_metadata_payload, pretty=True)
    _write_json(output_dir / "stations.json", stations_payload, pretty=False)
    return stations_payload


def _field_specs() -> dict[str, dict[str, Any]]:
    return {
        "slip_m": {
            "label": "Slip",
            "units": "m",
            "color_map": "viridis",
            "scale": "linear",
        },
        "delta_slip_m": {
            "label": "Delta slip",
            "units": "m",
            "color_map": "inferno",
            "scale": "linear",
        },
        "slip_rate_m_per_s": {
            "label": "Slip rate V",
            "units": "m/s",
            "color_map": "hot",
            "scale": "log",
        },
        "theta_s": {
            "label": "Theta",
            "units": "s",
            "color_map": "plasma",
            "scale": "log",
        },
        "a": {
            "label": "a",
            "units": "unitless",
            "color_map": "ylorred",
            "scale": "linear",
        },
        "b": {
            "label": "b",
            "units": "unitless",
            "color_map": "ylorred",
            "scale": "linear",
        },
        "D_c_m": {
            "label": "D_c",
            "units": "m",
            "color_map": "ylgnbu",
            "scale": "linear",
        },
        "a_minus_b": {
            "label": "a - b",
            "units": "unitless",
            "color_map": "rdbu",
            "scale": "symmetric",
        },
        "tau_elastic_pa": {
            "label": "Tau elastic",
            "units": "Pa",
            "color_map": "cividis",
            "scale": "linear",
        },
        "tau_rsf_pa": {
            "label": "Tau RSF",
            "units": "Pa",
            "color_map": "cividis",
            "scale": "linear",
        },
        "tau_residual_over_sigma_n": {
            "label": "RSF residual / sigma_n",
            "units": "unitless",
            "color_map": "rdbu",
            "scale": "symmetric",
        },
        "aging_residual": {
            "label": "Aging residual",
            "units": "unitless",
            "color_map": "rdbu",
            "scale": "symmetric",
        },
    }


def _static_field_specs() -> dict[str, dict[str, Any]]:
    return {
        "gnss_observability_m_per_m": {
            "label": "GNSS observability",
            "units": "m/m",
            "color_map": "cividis",
            "scale": "log",
        },
        "weighted_gnss_observability_m_per_m": {
            "label": "Weighted GNSS observability",
            "units": "m/m",
            "color_map": "ylorred",
            "scale": "log",
        },
    }


def _combined_field_specs(
    *,
    has_weighted_observability: bool,
) -> dict[str, dict[str, Any]]:
    specs = {
        "supported_delta_slip_m": {
            "label": "Supported delta slip",
            "units": "m",
            "color_map": "inferno",
            "scale": "linear",
        },
        "unsupported_delta_slip_m": {
            "label": "Under-supported delta slip",
            "units": "m",
            "color_map": "magma",
            "scale": "linear",
        },
        "delta_slip_over_support_m": {
            "label": "Delta slip / GNSS support",
            "units": "m",
            "color_map": "cividis",
            "scale": "log",
        },
    }
    if has_weighted_observability:
        specs.update(
            {
                "weighted_supported_delta_slip_m": {
                    "label": "Training-weighted supported delta slip",
                    "units": "m",
                    "color_map": "inferno",
                    "scale": "linear",
                },
                "weighted_unsupported_delta_slip_m": {
                    "label": "Training-weighted under-supported delta slip",
                    "units": "m",
                    "color_map": "magma",
                    "scale": "linear",
                },
                "delta_slip_over_weighted_support_m": {
                    "label": "Delta slip / weighted GNSS support",
                    "units": "m",
                    "color_map": "cividis",
                    "scale": "log",
                },
            }
        )
    return specs


def _load_station_sensitivity_map(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    payload = _load_json(path)
    if not isinstance(payload, list):
        return {}
    return {
        str(item.get("station", "")).strip().upper(): float(item.get("sensitivity", 0.0))
        for item in payload
        if str(item.get("station", "")).strip()
    }


def _compute_patch_observability_fields(
    *,
    green_dir: Path,
    station_ids: list[str],
    station_sensitivity_path: Path | None,
) -> dict[str, np.ndarray]:
    K_cd = np.load(green_dir / "K_cd_disp.npy", mmap_mode="r")
    if K_cd.shape[0] != 3 * len(station_ids):
        raise ValueError(
            f"K_cd row count {K_cd.shape[0]} does not match 3 x station count {len(station_ids)}."
        )

    station_component_norm = np.linalg.norm(
        np.asarray(K_cd, dtype=np.float64).reshape(len(station_ids), 3, -1),
        axis=1,
    )

    static_fields = {
        "gnss_observability_m_per_m": np.linalg.norm(station_component_norm, axis=0),
    }

    station_weights_map = _load_station_sensitivity_map(station_sensitivity_path)
    if station_weights_map:
        station_weights = np.asarray(
            [station_weights_map.get(station_id, 0.0) for station_id in station_ids],
            dtype=np.float64,
        )
        max_weight = float(station_weights.max())
        if max_weight > 0.0:
            station_weights = station_weights / max_weight
            static_fields["weighted_gnss_observability_m_per_m"] = np.linalg.norm(
                station_component_norm * station_weights[:, None],
                axis=0,
            )

    return static_fields


def _snapshot_dates(eval_table: pd.DataFrame, frequency: str) -> list[pd.Timestamp]:
    eval_times = pd.to_datetime(eval_table["date"])
    start = eval_times.min()
    end = eval_times.max()
    time_of_day = start - start.normalize()

    monthly = [timestamp + time_of_day for timestamp in pd.date_range(start.normalize(), end.normalize(), freq=frequency)]
    selected = [start, *monthly, end]

    uniq = []
    seen = set()
    for timestamp in selected:
        rounded = pd.Timestamp(timestamp)
        if rounded < start or rounded > end:
            continue
        key = rounded.isoformat()
        if key not in seen:
            seen.add(key)
            uniq.append(rounded)
    return uniq


def _predict_arrays(runtime: dict[str, Any], time_seconds: float) -> dict[str, np.ndarray]:
    model = runtime["model"]
    xi = runtime["xi"]
    eta = runtime["eta"]

    with torch.enable_grad():
        out = model(xi, eta, time_seconds)

    arrays = {
        key: value.detach().cpu().numpy().reshape(-1).astype(np.float64)
        for key, value in out.items()
    }
    arrays["a_minus_b"] = arrays["a"] - arrays["b"]

    sigma_n = float(model.sigma_n)
    arrays["tau_residual_over_sigma_n"] = (arrays["tau_elastic"] - arrays["tau_rsf"]) / sigma_n
    arrays["aging_residual"] = arrays["dtheta_dt"] - (
        1.0 - (arrays["V"] * arrays["theta"] / np.maximum(arrays["D_c"], 1e-12))
    )
    return arrays


def export_model_snapshots(
    *,
    spec: ModelExportSpec,
    green_dir: Path,
    output_dir: Path,
    snapshot_frequency: str,
) -> dict[str, Any]:
    runtime = load_trained_model(
        checkpoint_path=str(spec.checkpoint_path),
        green_dir=str(green_dir),
        device="cpu",
    )
    station_ids = [str(item).strip().upper() for item in np.load(green_dir / "station_ids.npy", allow_pickle=True).tolist()]
    eval_table = pd.read_csv(spec.eval_csv_path)
    dates = _snapshot_dates(eval_table, frequency=snapshot_frequency)
    reference_date = pd.Timestamp(runtime["config"]["reference_date"])
    field_specs = dict(_field_specs())
    model_dir = output_dir / "model_snapshots" / spec.key
    static_fields = _compute_patch_observability_fields(
        green_dir=green_dir,
        station_ids=station_ids,
        station_sensitivity_path=spec.station_sensitivity_path,
    )
    static_specs = _static_field_specs()
    for field_key in static_fields:
        field_specs[field_key] = static_specs[field_key]
    field_specs.update(
        _combined_field_specs(
            has_weighted_observability="weighted_gnss_observability_m_per_m" in static_fields
        )
    )

    field_ranges = {
        key: {"min": float("inf"), "max": float("-inf")}
        for key in field_specs
    }
    snapshots_manifest: list[dict[str, Any]] = []

    for field_key, values in static_fields.items():
        arr = np.asarray(values, dtype=np.float64)
        field_ranges[field_key]["min"] = float(arr.min())
        field_ranges[field_key]["max"] = float(arr.max())

    baseline_slip: np.ndarray | None = None
    for timestamp in dates:
        time_seconds = float((timestamp - reference_date).total_seconds())
        arrays = _predict_arrays(runtime, time_seconds)
        if baseline_slip is None:
            baseline_slip = arrays["s"].copy()
        delta_slip = arrays["s"] - baseline_slip
        delta_slip_abs = np.abs(delta_slip)
        observability = np.asarray(static_fields["gnss_observability_m_per_m"], dtype=np.float64)
        observability_norm = observability / max(float(observability.max()), 1e-12)
        support_floor = 0.08

        fields = {
            "slip_m": arrays["s"].tolist(),
            "delta_slip_m": delta_slip.tolist(),
            "slip_rate_m_per_s": arrays["V"].tolist(),
            "theta_s": arrays["theta"].tolist(),
            "a": arrays["a"].tolist(),
            "b": arrays["b"].tolist(),
            "D_c_m": arrays["D_c"].tolist(),
            "a_minus_b": arrays["a_minus_b"].tolist(),
            "tau_elastic_pa": arrays["tau_elastic"].tolist(),
            "tau_rsf_pa": arrays["tau_rsf"].tolist(),
            "tau_residual_over_sigma_n": arrays["tau_residual_over_sigma_n"].tolist(),
            "aging_residual": arrays["aging_residual"].tolist(),
            "supported_delta_slip_m": (delta_slip_abs * observability_norm).tolist(),
            "unsupported_delta_slip_m": (delta_slip_abs * (1.0 - observability_norm)).tolist(),
            "delta_slip_over_support_m": (
                delta_slip_abs / np.maximum(observability_norm, support_floor)
            ).tolist(),
        }
        if "weighted_gnss_observability_m_per_m" in static_fields:
            weighted_observability = np.asarray(
                static_fields["weighted_gnss_observability_m_per_m"], dtype=np.float64
            )
            weighted_observability_norm = weighted_observability / max(
                float(weighted_observability.max()), 1e-12
            )
            fields.update(
                {
                    "weighted_supported_delta_slip_m": (
                        delta_slip_abs * weighted_observability_norm
                    ).tolist(),
                    "weighted_unsupported_delta_slip_m": (
                        delta_slip_abs * (1.0 - weighted_observability_norm)
                    ).tolist(),
                    "delta_slip_over_weighted_support_m": (
                        delta_slip_abs
                        / np.maximum(weighted_observability_norm, support_floor)
                    ).tolist(),
                }
            )

        for field_key, values in fields.items():
            arr = np.asarray(values, dtype=np.float64)
            field_ranges[field_key]["min"] = min(field_ranges[field_key]["min"], float(arr.min()))
            field_ranges[field_key]["max"] = max(field_ranges[field_key]["max"], float(arr.max()))

        surface = {
            "station_ids": station_ids,
            "u_surface_m": np.asarray(arrays["u_surface"], dtype=np.float64).reshape(-1, 3).tolist(),
            "v_surface_m_per_s": np.asarray(arrays["v_surface"], dtype=np.float64).reshape(-1, 3).tolist(),
        }

        snapshot_payload = {
            "date": timestamp.isoformat(),
            "time_seconds": time_seconds,
            "fields": fields,
            "surface_predictions": surface,
        }
        date_key = timestamp.strftime("%Y-%m-%d")
        _write_json(model_dir / f"{date_key}.json", snapshot_payload, pretty=False)
        snapshots_manifest.append(
            {
                "date": timestamp.isoformat(),
                "date_key": date_key,
                "path": f"/data/model_snapshots/{spec.key}/{date_key}.json",
            }
        )

    payload = {
        "meta": {
            "model_key": spec.key,
            "model_label": spec.label,
            "checkpoint": str(spec.checkpoint_path.relative_to(REPO_ROOT)),
            "evaluation_csv": str(spec.eval_csv_path.relative_to(REPO_ROOT)),
            "history_csv": (
                str(spec.history_csv_path.relative_to(REPO_ROOT)) if spec.history_csv_path is not None else None
            ),
            "reference_date": reference_date.isoformat(),
            "snapshot_count": len(snapshots_manifest),
            "baseline_snapshot_date": snapshots_manifest[0]["date"] if snapshots_manifest else None,
            "patch_count": int(runtime["summary"]["Nc"]),
            "station_count": len(station_ids),
        },
        "fields": {
            key: {**spec, **field_ranges[key]}
            for key, spec in field_specs.items()
        },
        "static_fields": {
            key: np.asarray(values, dtype=np.float64).tolist()
            for key, values in static_fields.items()
        },
        "snapshots": snapshots_manifest,
    }
    _write_json(model_dir / "index.json", payload, pretty=True)
    return payload


def _summarize_evaluation_frame(evaluation: pd.DataFrame) -> dict[str, Any]:
    if evaluation.empty:
        return {
            "sample_count": 0,
            "date_range": {"start": None, "end": None},
            "median_rmse_mm": None,
            "p90_rmse_mm": None,
            "worst_rmse_mm": None,
        }
    return {
        "sample_count": int(len(evaluation)),
        "date_range": {
            "start": str(evaluation["date"].iloc[0]),
            "end": str(evaluation["date"].iloc[-1]),
        },
        "median_rmse_mm": float(evaluation["rmse_mm"].median()),
        "p90_rmse_mm": float(evaluation["rmse_mm"].quantile(0.9)),
        "worst_rmse_mm": float(evaluation["rmse_mm"].max()),
    }


def _summarize_history_frame(history: pd.DataFrame | None) -> dict[str, Any] | None:
    if history is None or history.empty:
        return None
    return {
        "step_count": int(len(history)),
        "final_total_loss": float(history["L_total"].iloc[-1]) if "L_total" in history.columns else None,
        "best_total_loss": float(history["L_total"].min()) if "L_total" in history.columns else None,
        "final_slip_max_m": float(history["slip_max"].iloc[-1]) if "slip_max" in history.columns else None,
    }


def export_model_catalog(
    *,
    checkpoints_root: Path,
    default_checkpoint_path: Path,
    default_eval_csv_path: Path,
    default_history_csv_path: Path,
    green_dir: Path,
    output_dir: Path,
    snapshot_frequency: str,
) -> dict[str, Any]:
    specs = _discover_model_specs(
        checkpoints_root=checkpoints_root,
        default_checkpoint_path=default_checkpoint_path,
        default_eval_csv_path=default_eval_csv_path,
        default_history_csv_path=default_history_csv_path,
    )

    models_payload: list[dict[str, Any]] = []
    default_model_key = _sanitize_model_key(default_checkpoint_path.parent.name)
    default_snapshot_payload: dict[str, Any] | None = None

    for spec in specs:
        snapshot_payload = export_model_snapshots(
            spec=spec,
            green_dir=green_dir,
            output_dir=output_dir,
            snapshot_frequency=snapshot_frequency,
        )
        station_sensitivity_export_path = None
        station_sensitivity_summary = None
        if spec.station_sensitivity_path is not None and spec.station_sensitivity_path.exists():
            ranking_payload = _load_json(spec.station_sensitivity_path)
            station_sensitivity_export_path = f"/data/model_station_sensitivity/{spec.key}.json"
            _write_json(
                output_dir / "model_station_sensitivity" / f"{spec.key}.json",
                ranking_payload,
                pretty=False,
            )
            if isinstance(ranking_payload, list) and ranking_payload:
                sensitivities = np.asarray(
                    [float(item.get("sensitivity", 0.0)) for item in ranking_payload],
                    dtype=np.float64,
                )
                station_sensitivity_summary = {
                    "station_count": int(len(ranking_payload)),
                    "max_sensitivity": float(sensitivities.max()),
                    "median_sensitivity": float(np.median(sensitivities)),
                    "top_station": str(ranking_payload[0].get("station", "")),
                }
        if spec.key == default_model_key:
            default_snapshot_payload = snapshot_payload
            _write_json(output_dir / "model_snapshots" / "index.json", snapshot_payload, pretty=True)

        evaluation = pd.read_csv(spec.eval_csv_path)
        history = pd.read_csv(spec.history_csv_path) if spec.history_csv_path is not None else None
        models_payload.append(
            {
                "key": spec.key,
                "label": spec.label,
                "checkpoint": str(spec.checkpoint_path.relative_to(REPO_ROOT)),
                "evaluation_csv": str(spec.eval_csv_path.relative_to(REPO_ROOT)),
                "history_csv": (
                    str(spec.history_csv_path.relative_to(REPO_ROOT)) if spec.history_csv_path is not None else None
                ),
                "snapshot_index_path": f"/data/model_snapshots/{spec.key}/index.json",
                "station_sensitivity_path": station_sensitivity_export_path,
                "snapshot_count": snapshot_payload["meta"]["snapshot_count"],
                "time_range": _summarize_evaluation_frame(evaluation)["date_range"],
                "metrics_summary": _summarize_evaluation_frame(evaluation),
                "history_summary": _summarize_history_frame(history),
                "station_sensitivity_summary": station_sensitivity_summary,
            }
        )

    payload = {
        "default_model_key": default_model_key,
        "models": models_payload,
    }
    _write_json(output_dir / "models" / "index.json", payload, pretty=True)
    if default_snapshot_payload is None:
        raise RuntimeError("Default model export failed; no snapshot payload generated.")
    return {
        "catalog": payload,
        "default_snapshot_payload": default_snapshot_payload,
    }


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, float)):
                record[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                record[key] = int(value)
            else:
                record[key] = value
        records.append(record)
    return records


def export_metrics(
    *,
    eval_csv_path: Path,
    history_csv_path: Path,
    station_sensitivity_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    evaluation = pd.read_csv(eval_csv_path)
    history = pd.read_csv(history_csv_path)

    evaluation_payload = {
        "summary": {
            "sample_count": int(len(evaluation)),
            "date_range": {
                "start": str(evaluation["date"].iloc[0]) if not evaluation.empty else None,
                "end": str(evaluation["date"].iloc[-1]) if not evaluation.empty else None,
            },
            "median_rmse_mm": float(evaluation["rmse_mm"].median()) if not evaluation.empty else None,
            "p90_rmse_mm": float(evaluation["rmse_mm"].quantile(0.9)) if not evaluation.empty else None,
            "worst_rmse_mm": float(evaluation["rmse_mm"].max()) if not evaluation.empty else None,
            "median_velocity_rmse_mm_per_day": (
                float(evaluation["velocity_rmse_mm_per_day"].median()) if not evaluation.empty else None
            ),
        },
        "records": _frame_records(evaluation),
    }

    history_payload = {
        "summary": {
            "step_count": int(len(history)),
            "final_total_loss": float(history["L_total"].iloc[-1]) if not history.empty else None,
            "best_total_loss": float(history["L_total"].min()) if not history.empty else None,
            "final_slip_max_m": float(history["slip_max"].iloc[-1]) if not history.empty else None,
            "final_a_minus_b_mean": (
                float(history["a_minus_b_mean"].iloc[-1]) if not history.empty else None
            ),
        },
        "records": _frame_records(history),
    }

    ranking_payload = _load_json(station_sensitivity_path)

    _write_json(output_dir / "metrics" / "evaluation.json", evaluation_payload, pretty=False)
    _write_json(output_dir / "metrics" / "training_history.json", history_payload, pretty=False)
    _write_json(output_dir / "metrics" / "station_sensitivity.json", ranking_payload, pretty=True)

    return {
        "evaluation": evaluation_payload["summary"],
        "training_history": history_payload["summary"],
    }


def export_manifest(
    *,
    output_dir: Path,
    fault_payload: dict[str, Any],
    stations_payload: dict[str, Any],
    snapshots_payload: dict[str, Any],
    model_catalog_payload: dict[str, Any],
    metrics_summary: dict[str, Any],
) -> None:
    manifest = {
        "app_title": "L'Aquila Fault Explorer",
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "paths": {
            "fault_geojson": "/data/fault.geojson",
            "fault_patches": "/data/fault_patches.json",
            "station_metadata": "/data/station_metadata.json",
            "stations": "/data/stations.json",
            "timeseries_dir": "/data/timeseries",
            "models_index": "/data/models/index.json",
            "snapshots_index": "/data/model_snapshots/index.json",
            "metrics": {
                "evaluation": "/data/metrics/evaluation.json",
                "training_history": "/data/metrics/training_history.json",
                "station_sensitivity": "/data/metrics/station_sensitivity.json",
            },
        },
        "summary": {
            "fault_patches": fault_payload["meta"]["patch_count"],
            "stations": stations_payload["meta"]["station_count"],
            "snapshots": snapshots_payload["meta"]["snapshot_count"],
            "models": len(model_catalog_payload["models"]),
            "coordinate_source": stations_payload["meta"]["coordinate_source"],
        },
        "metrics_summary": metrics_summary,
    }
    _write_json(output_dir / "manifest.json", manifest, pretty=True)


def main() -> None:
    args = _parse_args()
    output_dir = _as_path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    paths = _ensure_export_dirs(output_dir)

    fault_payload = export_fault_geometry(
        fault_geojson_path=REPO_ROOT / args.fault_geojson,
        green_dir=REPO_ROOT / args.green_dir,
        output_dir=paths.data_dir,
    )
    stations_payload = export_stations(
        data_dir=REPO_ROOT / args.data_dir,
        output_dir=paths.data_dir,
        station_ids_path=REPO_ROOT / args.green_dir / "station_ids.npy",
        fault_payload=fault_payload,
        stations_json_path=REPO_ROOT / args.stations_json if args.stations_json else None,
    )
    model_exports = export_model_catalog(
        checkpoints_root=REPO_ROOT / "checkpoints",
        default_checkpoint_path=REPO_ROOT / args.checkpoint,
        default_eval_csv_path=REPO_ROOT / args.eval_csv,
        default_history_csv_path=REPO_ROOT / args.history_csv,
        green_dir=REPO_ROOT / args.green_dir,
        output_dir=paths.data_dir,
        snapshot_frequency=args.snapshot_frequency,
    )
    snapshots_payload = model_exports["default_snapshot_payload"]
    metrics_summary = export_metrics(
        eval_csv_path=REPO_ROOT / args.eval_csv,
        history_csv_path=REPO_ROOT / args.history_csv,
        station_sensitivity_path=REPO_ROOT / args.station_sensitivity_json,
        output_dir=paths.data_dir,
    )
    export_manifest(
        output_dir=paths.data_dir,
        fault_payload=fault_payload,
        stations_payload=stations_payload,
        snapshots_payload=snapshots_payload,
        model_catalog_payload=model_exports["catalog"],
        metrics_summary=metrics_summary,
    )

    print(f"Export complete: {paths.data_dir}", flush=True)


if __name__ == "__main__":
    main()
