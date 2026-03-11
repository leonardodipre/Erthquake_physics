from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import PINNDataloader, list_markers_in_subdir
from predict import load_trained_model

SECONDS_PER_DAY = 86_400.0


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _masked_metrics(residual: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    valid = mask > 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "n_valid_components": 0.0,
            "rmse_m": 0.0,
            "mae_m": 0.0,
            "max_abs_m": 0.0,
        }

    residual_valid = residual[valid]
    return {
        "n_valid_components": float(n_valid),
        "rmse_m": float(np.sqrt(np.mean(residual_valid**2))),
        "mae_m": float(np.mean(np.abs(residual_valid))),
        "max_abs_m": float(np.max(np.abs(residual_valid))),
    }


def _resolve_excluded_markers(cfg: dict[str, Any], data_dir: str) -> list[str]:
    explicit = [str(marker).strip().upper() for marker in cfg.get("excluded_markers", ()) if str(marker).strip()]
    by_subdir = []
    subdir = cfg.get("excluded_markers_subdir")
    if subdir:
        by_subdir = list_markers_in_subdir(data_dir=data_dir, subdir=str(subdir))
    return sorted(dict.fromkeys(explicit + by_subdir))


def evaluate_gnss_misfit(
    checkpoint_path: str,
    green_dir: str = "green_out",
    data_dir: str = "dataset_scremato",
    device: torch.device | str | None = None,
    max_samples: int | None = None,
    output_csv: str | None = None,
) -> dict[str, Any]:
    runtime = load_trained_model(
        checkpoint_path=checkpoint_path,
        green_dir=green_dir,
        device=device,
    )
    cfg = runtime["config"]
    model = runtime["model"]
    xi = runtime["xi"]
    eta = runtime["eta"]
    device_t = runtime["device"]

    dataloader = PINNDataloader(
        data_dir=data_dir,
        station_ids_path=str(_as_path(green_dir) / "station_ids.npy"),
        time_ranges_data=cfg["time_ranges_data"],
        time_domain_physics=cfg["time_domain_physics"],
        reference_date=str(cfg["reference_date"]),
        seed=cfg.get("seed"),
        gap_min_days=float(cfg.get("gap_min_days", 0.5)),
        gap_max_days=float(cfg.get("gap_max_days", 1.5)),
        apply_robust_filter=bool(cfg.get("apply_robust_filter", True)),
        mad_scale=float(cfg.get("mad_scale", 1.4826)),
        mad_sigma_threshold=float(cfg.get("mad_sigma_threshold", 8.0)),
        inversion_sigma_threshold=float(cfg.get("inversion_sigma_threshold", 6.0)),
        inversion_ratio_min=float(cfg.get("inversion_ratio_min", 0.6)),
        inversion_cancellation_max=float(cfg.get("inversion_cancellation_max", 0.45)),
        slope_window_days=float(cfg.get("slope_window_days", 14.0)),
        slope_min_points=int(cfg.get("slope_min_points", 10)),
        excluded_markers=_resolve_excluded_markers(cfg=cfg, data_dir=data_dir),
    )

    data_indices = dataloader.data_indices
    if max_samples is not None:
        data_indices = data_indices[: int(max_samples)]

    rows: list[dict[str, Any]] = []
    weighted_position_sse = 0.0
    weighted_position_abs = 0.0
    total_valid_position = 0.0
    weighted_velocity_sse = 0.0
    weighted_velocity_abs = 0.0
    total_valid_velocity = 0.0

    print(
        f"Eval start | device={device_t} | checkpoint={checkpoint_path} | "
        f"samples={len(data_indices)}",
        flush=True,
    )

    for eval_idx, row_idx in enumerate(data_indices, start=1):
        t_seconds = float(dataloader.time_seconds[row_idx].item())
        timestamp = pd.Timestamp(dataloader.timestamps[row_idx])
        u_obs = dataloader.u_observed[row_idx].to(device=device_t, non_blocking=True)
        mask = dataloader.mask_data[row_idx].to(device=device_t, non_blocking=True)
        v_obs = dataloader.v_observed[row_idx].to(device=device_t, non_blocking=True)
        velocity_mask = dataloader.mask_velocity[row_idx].to(device=device_t, non_blocking=True)

        with torch.enable_grad():
            out = model(xi, eta, t_seconds)

        residual_position = ((out["u_surface"] - u_obs) * mask).detach().cpu().numpy()
        position_mask_np = mask.detach().cpu().numpy()
        position_metrics = _masked_metrics(residual=residual_position, mask=position_mask_np)

        residual_velocity = ((out["v_surface"] - v_obs) * velocity_mask).detach().cpu().numpy()
        velocity_mask_np = velocity_mask.detach().cpu().numpy()
        velocity_metrics = _masked_metrics(residual=residual_velocity, mask=velocity_mask_np)

        valid_position = position_metrics["n_valid_components"]
        weighted_position_sse += (position_metrics["rmse_m"] ** 2) * valid_position
        weighted_position_abs += position_metrics["mae_m"] * valid_position
        total_valid_position += valid_position

        valid_velocity = velocity_metrics["n_valid_components"]
        weighted_velocity_sse += (velocity_metrics["rmse_m"] ** 2) * valid_velocity
        weighted_velocity_abs += velocity_metrics["mae_m"] * valid_velocity
        total_valid_velocity += valid_velocity

        rows.append(
            {
                "index": int(row_idx),
                "date": timestamp.isoformat(),
                "time_seconds": t_seconds,
                "n_valid_components": int(valid_position),
                "n_valid_stations": int(valid_position // 3),
                "rmse_m": position_metrics["rmse_m"],
                "rmse_mm": position_metrics["rmse_m"] * 1e3,
                "mae_m": position_metrics["mae_m"],
                "mae_mm": position_metrics["mae_m"] * 1e3,
                "max_abs_m": position_metrics["max_abs_m"],
                "max_abs_mm": position_metrics["max_abs_m"] * 1e3,
                "n_valid_velocity_components": int(valid_velocity),
                "n_valid_velocity_stations": int(valid_velocity // 3),
                "velocity_rmse_m_per_s": velocity_metrics["rmse_m"],
                "velocity_rmse_mm_per_day": velocity_metrics["rmse_m"] * 1e3 * SECONDS_PER_DAY,
                "velocity_mae_m_per_s": velocity_metrics["mae_m"],
                "velocity_mae_mm_per_day": velocity_metrics["mae_m"] * 1e3 * SECONDS_PER_DAY,
                "velocity_max_abs_m_per_s": velocity_metrics["max_abs_m"],
                "velocity_max_abs_mm_per_day": velocity_metrics["max_abs_m"] * 1e3 * SECONDS_PER_DAY,
            }
        )

        if eval_idx == 1 or eval_idx % 250 == 0 or eval_idx == len(data_indices):
            print(
                f"Eval {eval_idx:5d}/{len(data_indices):5d} | "
                f"date={timestamp.date()} | "
                f"rmse_mm={position_metrics['rmse_m'] * 1e3:.3f} | "
                f"vel_rmse_mm_day={velocity_metrics['rmse_m'] * 1e3 * SECONDS_PER_DAY:.3f} | "
                f"valid_comp={int(valid_position)} | "
                f"valid_vel={int(valid_velocity)}",
                flush=True,
            )

    results_df = pd.DataFrame(rows)
    global_rmse_m = float(np.sqrt(weighted_position_sse / max(total_valid_position, 1.0)))
    global_mae_m = float(weighted_position_abs / max(total_valid_position, 1.0))
    global_velocity_rmse_m_per_s = float(
        np.sqrt(weighted_velocity_sse / max(total_valid_velocity, 1.0))
    )
    global_velocity_mae_m_per_s = float(
        weighted_velocity_abs / max(total_valid_velocity, 1.0)
    )

    summary = {
        "n_time_samples": int(len(results_df)),
        "time_range": (
            results_df["date"].iloc[0] if not results_df.empty else None,
            results_df["date"].iloc[-1] if not results_df.empty else None,
        ),
        "total_valid_components": int(total_valid_position),
        "total_valid_velocity_components": int(total_valid_velocity),
        "global_rmse_m": global_rmse_m,
        "global_rmse_mm": global_rmse_m * 1e3,
        "global_mae_m": global_mae_m,
        "global_mae_mm": global_mae_m * 1e3,
        "global_velocity_rmse_m_per_s": global_velocity_rmse_m_per_s,
        "global_velocity_rmse_mm_per_day": global_velocity_rmse_m_per_s * 1e3 * SECONDS_PER_DAY,
        "global_velocity_mae_m_per_s": global_velocity_mae_m_per_s,
        "global_velocity_mae_mm_per_day": global_velocity_mae_m_per_s * 1e3 * SECONDS_PER_DAY,
        "median_time_rmse_mm": float(results_df["rmse_mm"].median()) if not results_df.empty else 0.0,
        "p90_time_rmse_mm": float(results_df["rmse_mm"].quantile(0.90)) if not results_df.empty else 0.0,
        "worst_time_rmse_mm": float(results_df["rmse_mm"].max()) if not results_df.empty else 0.0,
        "median_time_velocity_rmse_mm_per_day": (
            float(results_df["velocity_rmse_mm_per_day"].median()) if not results_df.empty else 0.0
        ),
        "p90_time_velocity_rmse_mm_per_day": (
            float(results_df["velocity_rmse_mm_per_day"].quantile(0.90)) if not results_df.empty else 0.0
        ),
        "worst_time_velocity_rmse_mm_per_day": (
            float(results_df["velocity_rmse_mm_per_day"].max()) if not results_df.empty else 0.0
        ),
        "mean_valid_stations": float(results_df["n_valid_stations"].mean()) if not results_df.empty else 0.0,
        "median_valid_stations": float(results_df["n_valid_stations"].median()) if not results_df.empty else 0.0,
        "mean_valid_velocity_stations": (
            float(results_df["n_valid_velocity_stations"].mean()) if not results_df.empty else 0.0
        ),
        "median_valid_velocity_stations": (
            float(results_df["n_valid_velocity_stations"].median()) if not results_df.empty else 0.0
        ),
    }

    if output_csv is not None:
        output_path = _as_path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)

    return {
        "summary": summary,
        "table": results_df,
        "config": cfg,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate GNSS misfit for a trained model")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--data-dir", default="dataset_scremato")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--out-csv", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    evaluation = evaluate_gnss_misfit(
        checkpoint_path=args.checkpoint,
        green_dir=args.green_dir,
        data_dir=args.data_dir,
        device=args.device,
        max_samples=args.max_samples,
        output_csv=args.out_csv,
    )

    for key, value in evaluation["summary"].items():
        print(f"{key}: {value}", flush=True)
    if args.out_csv is not None:
        print(f"eval_table_saved: {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
