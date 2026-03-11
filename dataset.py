from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

DEFAULT_DATA_SUBDIRS = ("modified", "accepted", "acc_test")
DEFAULT_OBS_COLS = ("E_clean", "N_clean", "U_clean")
FALLBACK_OBS_COLS = ("E", "N", "U")


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _is_year_like(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


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
    ) -> None:
        self.data_dir = _as_path(data_dir)
        self.station_order = np.load(_as_path(station_ids_path), allow_pickle=True).tolist()
        self.reference_date = pd.Timestamp(reference_date)
        self.rng = np.random.default_rng(seed)
        self.data_subdirs = tuple(data_subdirs)

        self.station_files = self._select_station_files()
        self.timestamps, u_obs, masks = self._build_dense_observation_matrices()
        self.u_observed = torch.from_numpy(u_obs)
        self.mask_data = torch.from_numpy(masks)
        self.time_seconds = torch.tensor(
            (self.timestamps - self.reference_date).total_seconds().to_numpy(),
            dtype=torch.float32,
        )
        self.data_indices = self._build_data_indices(time_ranges_data=time_ranges_data)
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
                marker = csv_path.name.split("_")[1]
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

    def _build_dense_observation_matrices(self) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
        station_frames: dict[str, pd.DataFrame] = {}
        date_series = []
        for marker in self.station_order:
            frame = self._read_station_frame(self.station_files[marker])
            station_frames[marker] = frame
            date_series.append(frame["date"])

        timestamps = pd.DatetimeIndex(sorted(set(pd.concat(date_series).tolist())))
        n_times = len(timestamps)
        n_obs = 3 * len(self.station_order)
        u_obs = np.zeros((n_times, n_obs), dtype=np.float32)
        masks = np.zeros((n_times, n_obs), dtype=np.float32)
        time_to_row = {ts: idx for idx, ts in enumerate(timestamps)}

        for station_idx, marker in enumerate(self.station_order):
            frame = station_frames[marker]
            row_idx = np.fromiter((time_to_row[ts] for ts in frame["date"]), dtype=np.int64)
            values = frame.loc[:, ["E", "N", "U"]].to_numpy(dtype=np.float32, copy=True)
            valid = np.isfinite(values).astype(np.float32)
            values = np.nan_to_num(values, copy=False)
            col_slice = slice(3 * station_idx, 3 * station_idx + 3)
            u_obs[row_idx, col_slice] = values
            masks[row_idx, col_slice] = valid
        return timestamps, u_obs, masks

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
        has_data = self.mask_data.sum(dim=1).numpy() > 0
        return np.flatnonzero(keep & has_data)

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
            batch.append(
                {
                    "t_seconds": float(self.time_seconds[row].item()),
                    "u_observed": self.u_observed[row].clone(),
                    "mask_data": self.mask_data[row].clone(),
                }
            )
        return batch

    def sample_collocation_batch(self, batch_size: int = 64) -> list[float]:
        return self.rng.uniform(self.t_phys_min, self.t_phys_max, size=batch_size).tolist()
