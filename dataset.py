from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

DEFAULT_DATA_SUBDIRS = ("modified", "accepted", "acc_test")
DEFAULT_OBS_COLS = ("E_clean", "N_clean", "U_clean")
FALLBACK_OBS_COLS = ("E", "N", "U")
COMPONENTS = ("E", "N", "U")
SECONDS_PER_DAY = 86_400.0
_DEG_TO_KM_LAT = 111.32
_DEG_TO_KM_LON_AT_42 = 82.73  # 111.32 * cos(42°)


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _is_year_like(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _normalize_time_ranges(
    time_ranges_data: Sequence[tuple[Any, Any]],
) -> tuple[tuple[Any, Any], ...]:
    normalized: list[tuple[Any, Any]] = []
    for idx, item in enumerate(time_ranges_data):
        if isinstance(item, (str, bytes)):
            raise ValueError(
                "time_ranges_data must contain (start, end) pairs, "
                f"but item {idx} is a string: {item!r}."
            )
        try:
            start, end = item
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "time_ranges_data must be a sequence of (start, end) pairs; "
                f"got item {idx}: {item!r}. If you meant a year interval, use "
                "(2019, 2025), not (2019-2025)."
            ) from exc
        normalized.append((start, end))
    return tuple(normalized)


def extract_marker_from_filename(path_like: str | Path) -> str:
    path = _as_path(path_like)
    parts = path.stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"Cannot extract marker from filename: {path}")
    return parts[1].strip().upper()


def list_markers_in_subdir(data_dir: str | Path, subdir: str) -> list[str]:
    folder = _as_path(data_dir) / subdir
    if not folder.exists():
        return []
    markers = [extract_marker_from_filename(csv_path) for csv_path in sorted(folder.glob("*.csv"))]
    return sorted(dict.fromkeys(markers))


def compute_station_geo_weights(
    station_ids_path: str,
    stations_json_path: str,
    k: int = 5,
    alpha: float = 0.5,
    clip_min: float = 0.5,
    clip_max: float = 2.0,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Compute geometric density-based weights for GNSS stations.

    Returns
    -------
    weights_component : ndarray, shape (n_stations * 3,)
        Per-component weights (same weight for E/N/U of each station),
        normalized so that mean == 1.
    diagnostics : list[dict]
        Per-station info: marker, lat, lon, mean_knn_dist, raw_weight, weight.
    """
    import json

    station_order = np.load(station_ids_path, allow_pickle=True).tolist()
    station_order = [str(m).strip().upper() for m in station_order]

    with open(stations_json_path) as f:
        catalog = json.load(f)
    coord_map: dict[str, tuple[float, float]] = {}
    for s in catalog["stations"]:
        marker = str(s["marker"]).strip().upper()
        coord_map[marker] = (float(s["aprioriNorth"]), float(s["aprioriEast"]))

    n = len(station_order)
    coords_km = np.zeros((n, 2), dtype=np.float64)
    for i, marker in enumerate(station_order):
        lat, lon = coord_map[marker]
        coords_km[i, 0] = lat * _DEG_TO_KM_LAT
        coords_km[i, 1] = lon * _DEG_TO_KM_LON_AT_42

    # Pairwise distances
    from scipy.spatial import cKDTree
    tree = cKDTree(coords_km)
    dists, _ = tree.query(coords_km, k=k + 1)  # includes self at dist=0
    mean_knn_dist = dists[:, 1:].mean(axis=1)  # exclude self

    median_d = np.median(mean_knn_dist)
    if median_d <= 0:
        median_d = 1.0

    raw_weights = (mean_knn_dist / median_d) ** alpha
    clipped = np.clip(raw_weights, clip_min, clip_max)
    clipped /= clipped.mean()  # normalize to mean=1

    # Expand to per-component (E/N/U share same weight)
    weights_component = np.repeat(clipped, 3).astype(np.float32)

    diagnostics = []
    for i, marker in enumerate(station_order):
        lat, lon = coord_map[marker]
        diagnostics.append({
            "station": marker,
            "lat": lat,
            "lon": lon,
            "mean_knn_dist_km": float(mean_knn_dist[i]),
            "raw_weight": float(raw_weights[i]),
            "weight": float(clipped[i]),
        })

    return weights_component, diagnostics


class PINNDataloader:
    def __init__(
        self,
        data_dir: str,
        station_ids_path: str,
        time_ranges_data: Sequence[tuple[Any, Any]],
        time_domain_physics: tuple[Any, Any],
        reference_date: str = "2000-01-01",
        data_subdirs: Sequence[str] = DEFAULT_DATA_SUBDIRS,
        seed: int | None = None,
        gap_min_days: float = 0.5,
        gap_max_days: float = 1.5,
        apply_robust_filter: bool = True,
        mad_scale: float = 1.4826,
        mad_sigma_threshold: float = 8.0,
        inversion_sigma_threshold: float = 6.0,
        inversion_ratio_min: float = 0.6,
        inversion_cancellation_max: float = 0.45,
        slope_window_days: float = 14.0,
        slope_min_points: int = 10,
        excluded_markers: Sequence[str] | None = None,
        max_station_jump_m: float | None = None,
        savgol_window: int = 0,
        savgol_polyorder: int = 3,
    ) -> None:
        self.data_dir = _as_path(data_dir)
        self.time_ranges_data = _normalize_time_ranges(time_ranges_data)
        self.station_order = np.load(_as_path(station_ids_path), allow_pickle=True).tolist()
        self.station_order = [str(marker).strip().upper() for marker in self.station_order]
        self.station_index = {marker: idx for idx, marker in enumerate(self.station_order)}
        self.reference_date = pd.Timestamp(reference_date)
        self.rng = np.random.default_rng(seed)
        self.data_subdirs = tuple(data_subdirs)
        self.excluded_markers = sorted(
            marker for marker in {str(m).strip().upper() for m in (excluded_markers or ())} if marker
        )

        self.gap_min_days = float(gap_min_days)
        self.gap_max_days = float(gap_max_days)
        if self.gap_min_days <= 0.0:
            raise ValueError("gap_min_days must be strictly positive.")
        if self.gap_max_days < self.gap_min_days:
            raise ValueError("gap_max_days must be >= gap_min_days.")

        self.apply_robust_filter = bool(apply_robust_filter)
        self.mad_scale = float(mad_scale)
        self.mad_sigma_threshold = float(mad_sigma_threshold)
        self.inversion_sigma_threshold = float(inversion_sigma_threshold)
        self.inversion_ratio_min = float(inversion_ratio_min)
        self.inversion_cancellation_max = float(inversion_cancellation_max)
        self.slope_window_days = float(slope_window_days)
        self.slope_min_points = int(slope_min_points)
        if self.slope_window_days <= 0.0:
            raise ValueError("slope_window_days must be strictly positive.")
        if self.slope_min_points < 2:
            raise ValueError("slope_min_points must be >= 2.")

        self.savgol_window = int(savgol_window)
        if self.savgol_window > 0 and self.savgol_window % 2 == 0:
            self.savgol_window += 1
        self.savgol_polyorder = int(savgol_polyorder)
        if self.savgol_window > 0 and self.savgol_polyorder >= self.savgol_window:
            raise ValueError("savgol_polyorder must be less than savgol_window.")

        self.station_files = self._select_station_files()
        (
            self.timestamps,
            u_obs,
            position_masks,
            v_obs,
            velocity_masks,
            filter_report,
            self._station_quality,
        ) = self._build_dense_observation_matrices()
        self.u_observed = torch.from_numpy(u_obs)
        self.mask_data = torch.from_numpy(position_masks)
        self.v_observed = torch.from_numpy(v_obs)
        self.mask_velocity = torch.from_numpy(velocity_masks)
        self.filter_report = filter_report

        if max_station_jump_m is not None:
            threshold = float(max_station_jump_m)
            auto_excluded = [
                marker for marker, q in self._station_quality.items()
                if q["max_jump_m"] > threshold and marker not in set(self.excluded_markers)
            ]
            if auto_excluded:
                self.excluded_markers = sorted(set(self.excluded_markers) | set(auto_excluded))
                print(
                    f"Auto-excluded {len(auto_excluded)} stations (max_jump > {threshold:.3f} m): "
                    + ", ".join(auto_excluded),
                    flush=True,
                )

        self.training_component_mask = self.build_component_mask(
            [marker for marker in self.station_order if marker not in set(self.excluded_markers)]
        )
        self.holdout_component_mask = self.build_component_mask(self.excluded_markers)
        self.filter_report["n_excluded_stations"] = len(self.excluded_markers)
        self.filter_report["excluded_markers"] = list(self.excluded_markers)
        self.time_seconds = torch.tensor(
            (self.timestamps - self.reference_date).total_seconds().to_numpy(),
            dtype=torch.float32,
        )
        self.data_indices = self._build_data_indices(time_ranges_data=self.time_ranges_data)
        if len(self.data_indices) == 0:
            raise ValueError("No GNSS timestamps fall inside the requested data ranges.")

        self.t_phys_min = self._to_seconds(time_domain_physics[0], is_end=False)
        self.t_phys_max = self._to_seconds(time_domain_physics[1], is_end=True)
        if self.t_phys_max <= self.t_phys_min:
            raise ValueError("time_domain_physics must define a non-empty interval.")

    def _select_station_files(self) -> dict[str, str]:
        selected: dict[str, str] = {}
        station_set = set(self.station_order)
        for subdir in self.data_subdirs:
            folder = self.data_dir / subdir
            if not folder.exists():
                continue
            for csv_path in sorted(folder.glob("*.csv")):
                marker = extract_marker_from_filename(csv_path)
                if marker in station_set and marker not in selected:
                    selected[marker] = str(csv_path)
        missing = sorted(station_set - set(selected))
        if missing:
            raise FileNotFoundError(
                f"Missing GNSS files for {len(missing)} stations, e.g. {missing[:5]}"
            )
        return selected

    def _read_station_frame(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        if all(col in df.columns for col in DEFAULT_OBS_COLS):
            component_cols = DEFAULT_OBS_COLS
        elif all(col in df.columns for col in FALLBACK_OBS_COLS):
            component_cols = FALLBACK_OBS_COLS
        else:
            raise ValueError(f"Unexpected columns in {csv_path}")

        out = df.loc[:, ["date", *component_cols]].copy()
        out.columns = ["date", "E", "N", "U"]
        out["date"] = pd.to_datetime(out["date"])
        out = out.dropna(subset=["date"]).drop_duplicates(subset="date", keep="last")
        out = out.sort_values("date", kind="stable").reset_index(drop=True)
        return out

    def _robust_center_sigma(self, values: np.ndarray) -> tuple[float, float]:
        if values.size == 0:
            return 0.0, 1.0
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        sigma = self.mad_scale * mad
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.std(values))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        return center, sigma

    def _windowed_slopes_m_per_s(
        self,
        t_days: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if values.ndim != 2 or values.shape[1] != len(COMPONENTS):
            raise ValueError("values must have shape (n_samples, 3).")

        half_window_days = 0.5 * self.slope_window_days
        left_idx = np.searchsorted(t_days, t_days - half_window_days, side="left")
        right_idx = np.searchsorted(t_days, t_days + half_window_days, side="right")
        n_window = right_idx - left_idx

        prefix_t = np.concatenate(([0.0], np.cumsum(t_days, dtype=np.float64)))
        prefix_t2 = np.concatenate(([0.0], np.cumsum(t_days * t_days, dtype=np.float64)))
        sum_t = prefix_t[right_idx] - prefix_t[left_idx]
        sum_t2 = prefix_t2[right_idx] - prefix_t2[left_idx]

        denom = n_window * sum_t2 - sum_t * sum_t
        valid = (n_window >= self.slope_min_points) & (np.abs(denom) > 1e-20)
        safe_denom = np.where(valid, denom, 1.0)

        slopes = np.full(values.shape, np.nan, dtype=np.float64)
        for comp_idx in range(values.shape[1]):
            comp = values[:, comp_idx].astype(np.float64, copy=False)
            prefix_comp = np.concatenate(([0.0], np.cumsum(comp, dtype=np.float64)))
            prefix_t_comp = np.concatenate(([0.0], np.cumsum(t_days * comp, dtype=np.float64)))

            sum_comp = prefix_comp[right_idx] - prefix_comp[left_idx]
            sum_t_comp = prefix_t_comp[right_idx] - prefix_t_comp[left_idx]

            slope_m_per_day = (n_window * sum_t_comp - sum_t * sum_comp) / safe_denom
            slopes[:, comp_idx] = np.where(valid, slope_m_per_day / SECONDS_PER_DAY, np.nan)

        return slopes.astype(np.float32), valid

    def _robust_flags(self, values: np.ndarray, valid_rows: np.ndarray) -> np.ndarray:
        flags = np.zeros(values.shape[0], dtype=bool)
        if not self.apply_robust_filter or not valid_rows.any():
            return flags

        valid_values = values[valid_rows]
        center, sigma = self._robust_center_sigma(valid_values)
        centered = values - center

        flags |= valid_rows & (np.abs(centered) > self.mad_sigma_threshold * sigma)

        pair_idx = np.flatnonzero(valid_rows[:-1] & valid_rows[1:])
        if pair_idx.size == 0:
            return flags

        left = centered[pair_idx]
        right = centered[pair_idx + 1]
        abs_left = np.abs(left)
        abs_right = np.abs(right)
        max_abs = np.maximum(abs_left, abs_right)
        min_abs = np.minimum(abs_left, abs_right)
        ratio = min_abs / np.maximum(max_abs, 1e-12)
        cancellation = np.abs(left + right) / np.maximum(max_abs, 1e-12)

        inverted = (
            (left * right < 0.0)
            & (abs_left >= self.inversion_sigma_threshold * sigma)
            & (abs_right >= self.inversion_sigma_threshold * sigma)
            & (ratio >= self.inversion_ratio_min)
            & (cancellation <= self.inversion_cancellation_max)
        )
        flagged_pairs = pair_idx[inverted]
        flags[flagged_pairs] = True
        flags[flagged_pairs + 1] = True
        return flags

    def _apply_savgol_positions(
        self,
        t_days: np.ndarray,
        values: np.ndarray,
        finite_mask: np.ndarray,
    ) -> np.ndarray:
        """Smooth position time series with Savitzky-Golay filter.

        Interpolates over gaps, applies SG, then keeps only originally
        finite values so masked entries stay at zero.
        """
        from scipy.signal import savgol_filter

        result = values.copy()
        window = self.savgol_window
        polyorder = self.savgol_polyorder

        for comp_idx in range(values.shape[1]):
            finite = finite_mask[:, comp_idx]
            if finite.sum() < window:
                continue
            t_fin = t_days[finite]
            v_fin = values[finite, comp_idx]
            interp_vals = np.interp(t_days, t_fin, v_fin)
            smoothed = savgol_filter(interp_vals, window, polyorder)
            result[finite, comp_idx] = smoothed[finite]

        return result

    def _prepare_station_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
        out = frame.copy()
        values = out.loc[:, list(COMPONENTS)].to_numpy(dtype=np.float64, copy=True)
        finite_positions = np.isfinite(values)
        values = np.nan_to_num(values, nan=0.0)

        t_days = (
            (out["date"] - out["date"].iloc[0]).dt.total_seconds().to_numpy(dtype=np.float64)
            / SECONDS_PER_DAY
        )

        if self.savgol_window > 0:
            values = self._apply_savgol_positions(t_days, values, finite_positions)

        slopes_m_per_s, slope_window_valid = self._windowed_slopes_m_per_s(t_days=t_days, values=values)
        velocity_base_mask = slope_window_valid[:, None] & np.isfinite(slopes_m_per_s)

        robust_flags = np.zeros_like(velocity_base_mask, dtype=bool)
        for comp_idx in range(len(COMPONENTS)):
            robust_flags[:, comp_idx] = self._robust_flags(
                values=slopes_m_per_s[:, comp_idx],
                valid_rows=velocity_base_mask[:, comp_idx],
            )

        position_mask = finite_positions & ~robust_flags
        velocity_mask = velocity_base_mask & ~robust_flags

        for comp_idx, component in enumerate(COMPONENTS):
            out[component] = values[:, comp_idx].astype(np.float32)
            out[f"v{component}_slope"] = np.nan_to_num(
                slopes_m_per_s[:, comp_idx], nan=0.0
            ).astype(np.float32)
            out[f"mask_pos_{component}"] = position_mask[:, comp_idx].astype(np.float32)
            out[f"mask_vel_{component}"] = velocity_mask[:, comp_idx].astype(np.float32)
            out[f"flag_robust_{component}"] = robust_flags[:, comp_idx].astype(np.float32)

        daily_jumps = np.abs(np.diff(values, axis=0))
        max_jump_m = float(daily_jumps.max()) if daily_jumps.size > 0 else 0.0

        report = {
            "position_total_components": int(finite_positions.sum()),
            "position_valid_components": int(position_mask.sum()),
            "velocity_base_components": int(velocity_base_mask.sum()),
            "velocity_valid_components": int(velocity_mask.sum()),
            "robust_excluded_components": int(robust_flags.sum()),
            "max_jump_m": max_jump_m,
            "max_abs_position_m": float(np.abs(values).max()) if values.size > 0 else 0.0,
        }
        return out, report

    def _build_dense_observation_matrices(
        self,
    ) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int], dict[str, dict[str, Any]]]:
        station_frames: dict[str, pd.DataFrame] = {}
        filter_reports: list[dict[str, int]] = []
        per_station_quality: dict[str, dict[str, Any]] = {}
        date_series = []
        for marker in self.station_order:
            raw_frame = self._read_station_frame(self.station_files[marker])
            processed_frame, report = self._prepare_station_frame(raw_frame)
            station_frames[marker] = processed_frame
            filter_reports.append(report)
            per_station_quality[marker] = {
                "max_jump_m": report["max_jump_m"],
                "max_abs_position_m": report["max_abs_position_m"],
                "n_robust_excluded": report["robust_excluded_components"],
                "n_points": len(processed_frame),
            }
            date_series.append(processed_frame["date"])

        timestamps = pd.DatetimeIndex(sorted(set(pd.concat(date_series).tolist())))
        n_times = len(timestamps)
        n_obs = 3 * len(self.station_order)
        u_obs = np.zeros((n_times, n_obs), dtype=np.float32)
        position_masks = np.zeros((n_times, n_obs), dtype=np.float32)
        v_obs = np.zeros((n_times, n_obs), dtype=np.float32)
        velocity_masks = np.zeros((n_times, n_obs), dtype=np.float32)
        time_to_row = {ts: idx for idx, ts in enumerate(timestamps)}

        for station_idx, marker in enumerate(self.station_order):
            frame = station_frames[marker]
            row_idx = np.fromiter((time_to_row[ts] for ts in frame["date"]), dtype=np.int64)
            component_values = frame.loc[:, list(COMPONENTS)].to_numpy(dtype=np.float32, copy=True)
            slope_values = frame.loc[:, [f"v{comp}_slope" for comp in COMPONENTS]].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            pos_mask = frame.loc[:, [f"mask_pos_{comp}" for comp in COMPONENTS]].to_numpy(
                dtype=np.float32,
                copy=True,
            )
            vel_mask = frame.loc[:, [f"mask_vel_{comp}" for comp in COMPONENTS]].to_numpy(
                dtype=np.float32,
                copy=True,
            )

            col_slice = slice(3 * station_idx, 3 * station_idx + 3)
            u_obs[row_idx, col_slice] = component_values
            position_masks[row_idx, col_slice] = pos_mask
            v_obs[row_idx, col_slice] = slope_values
            velocity_masks[row_idx, col_slice] = vel_mask

        report = {
            "n_stations": len(self.station_order),
            "n_timestamps": n_times,
            "position_total_components": int(sum(r["position_total_components"] for r in filter_reports)),
            "position_valid_components": int(sum(r["position_valid_components"] for r in filter_reports)),
            "velocity_base_components": int(sum(r["velocity_base_components"] for r in filter_reports)),
            "velocity_valid_components": int(sum(r["velocity_valid_components"] for r in filter_reports)),
            "robust_excluded_components": int(sum(r["robust_excluded_components"] for r in filter_reports)),
        }
        if report["position_total_components"] > 0:
            report["position_valid_fraction"] = (
                report["position_valid_components"] / report["position_total_components"]
            )
        else:
            report["position_valid_fraction"] = 0.0
        if report["velocity_base_components"] > 0:
            report["velocity_valid_fraction"] = (
                report["velocity_valid_components"] / report["velocity_base_components"]
            )
            report["robust_excluded_fraction"] = (
                report["robust_excluded_components"] / report["velocity_base_components"]
            )
        else:
            report["velocity_valid_fraction"] = 0.0
            report["robust_excluded_fraction"] = 0.0

        return timestamps, u_obs, position_masks, v_obs, velocity_masks, report, per_station_quality

    def _build_data_indices(self, time_ranges_data: Sequence[tuple[Any, Any]]) -> np.ndarray:
        time_seconds = self.time_seconds.numpy()
        keep = np.zeros(time_seconds.shape[0], dtype=bool)
        for start, end in time_ranges_data:
            start_sec = self._to_seconds(start, is_end=False)
            end_sec = self._to_seconds(end, is_end=True)
            if _is_year_like(end):
                keep |= (time_seconds >= start_sec) & (time_seconds < end_sec)
            else:
                keep |= (time_seconds >= start_sec) & (time_seconds <= end_sec)
        masked_position = self.mask_data * self.training_component_mask.unsqueeze(0)
        masked_velocity = self.mask_velocity * self.training_component_mask.unsqueeze(0)
        has_position = masked_position.sum(dim=1).numpy() > 0
        has_velocity = masked_velocity.sum(dim=1).numpy() > 0
        return np.flatnonzero(keep & (has_position | has_velocity))

    def _to_timestamp(self, year_or_date: Any, is_end: bool = False) -> pd.Timestamp:
        if _is_year_like(year_or_date):
            year = int(year_or_date)
            return pd.Timestamp(f"{year + int(is_end)}-01-01")
        return pd.Timestamp(year_or_date)

    def _to_seconds(self, year_or_date: Any, is_end: bool = False) -> float:
        timestamp = self._to_timestamp(year_or_date, is_end=is_end)
        return float((timestamp - self.reference_date).total_seconds())

    def sample_data_batch(self, batch_size: int = 32) -> list[dict[str, Any]]:
        replace = batch_size > len(self.data_indices)
        sampled_rows = self.rng.choice(self.data_indices, size=batch_size, replace=replace)
        batch = []
        for row in np.atleast_1d(sampled_rows):
            full_position_mask = self.mask_data[row].clone()
            full_velocity_mask = self.mask_velocity[row].clone()
            batch.append(
                {
                    "t_seconds": float(self.time_seconds[row].item()),
                    "u_observed": self.u_observed[row].clone(),
                    "mask_data": full_position_mask * self.training_component_mask,
                    "v_observed": self.v_observed[row].clone(),
                    "mask_velocity": full_velocity_mask * self.training_component_mask,
                    "mask_data_full": full_position_mask,
                    "mask_velocity_full": full_velocity_mask,
                }
            )
        return batch

    def sample_collocation_batch(self, batch_size: int = 64) -> list[float]:
        return self.rng.uniform(self.t_phys_min, self.t_phys_max, size=batch_size).tolist()

    def station_quality_report(self) -> list[dict[str, Any]]:
        """Return per-station quality info sorted by max_jump_m (descending)."""
        rows = [
            {"marker": marker, **quality}
            for marker, quality in self._station_quality.items()
        ]
        rows.sort(key=lambda r: r["max_jump_m"], reverse=True)
        return rows

    def build_component_mask(self, markers: Sequence[str]) -> torch.Tensor:
        mask = torch.zeros(3 * len(self.station_order), dtype=torch.float32)
        for raw_marker in markers:
            marker = str(raw_marker).strip().upper()
            station_idx = self.station_index.get(marker)
            if station_idx is None:
                continue
            mask[3 * station_idx : 3 * station_idx + 3] = 1.0
        return mask
