"""Evaluate PINN predictions on holdout stations excluded from training."""
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


def evaluate_holdout_stations(
    checkpoint_path: str,
    green_dir: str = "green_out",
    data_dir: str = "dataset_scremato",
    device: torch.device | str | None = None,
    markers_subdir: str = "acc_test",
    days: int | None = None,
    max_samples: int | None = None,
    output_dir: str | None = None,
    make_plots: bool = False,
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

    holdout_markers = list_markers_in_subdir(data_dir=data_dir, subdir=markers_subdir)
    if not holdout_markers:
        return {
            "summary": {"n_time_samples": 0, "global_holdout_rmse_mm": 0.0},
            "table": pd.DataFrame(),
            "config": cfg,
        }

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

    holdout_mask = dataloader.build_component_mask(holdout_markers)

    data_indices = dataloader.data_indices
    if max_samples is not None:
        data_indices = data_indices[:int(max_samples)]

    rows: list[dict[str, Any]] = []
    weighted_sse = 0.0
    weighted_abs = 0.0
    total_valid = 0.0

    for row_idx in data_indices:
        t_seconds = float(dataloader.time_seconds[row_idx].item())
        timestamp = pd.Timestamp(dataloader.timestamps[row_idx])
        u_obs = dataloader.u_observed[row_idx].to(device=device_t, non_blocking=True)
        mask_full = dataloader.mask_data[row_idx].to(device=device_t, non_blocking=True)
        holdout_mask_device = holdout_mask.to(device=device_t, non_blocking=True)
        mask_holdout = mask_full * holdout_mask_device

        with torch.enable_grad():
            out = model(xi, eta, t_seconds)

        residual = ((out["u_surface"] - u_obs) * mask_holdout).detach().cpu().numpy()
        mask_np = mask_holdout.detach().cpu().numpy()
        metrics = _masked_metrics(residual=residual, mask=mask_np)

        n_valid = metrics["n_valid_components"]
        weighted_sse += (metrics["rmse_m"] ** 2) * n_valid
        weighted_abs += metrics["mae_m"] * n_valid
        total_valid += n_valid

        rows.append({
            "index": int(row_idx),
            "date": timestamp.isoformat(),
            "time_seconds": t_seconds,
            "n_valid_components": int(n_valid),
            "rmse_m": metrics["rmse_m"],
            "rmse_mm": metrics["rmse_m"] * 1e3,
            "mae_m": metrics["mae_m"],
            "mae_mm": metrics["mae_m"] * 1e3,
        })

    results_df = pd.DataFrame(rows)
    global_rmse_m = float(np.sqrt(weighted_sse / max(total_valid, 1.0)))
    global_mae_m = float(weighted_abs / max(total_valid, 1.0))

    summary = {
        "n_time_samples": len(results_df),
        "n_holdout_stations": len(holdout_markers),
        "holdout_markers": holdout_markers,
        "total_valid_components": int(total_valid),
        "global_holdout_rmse_m": global_rmse_m,
        "global_holdout_rmse_mm": global_rmse_m * 1e3,
        "global_holdout_mae_m": global_mae_m,
        "global_holdout_mae_mm": global_mae_m * 1e3,
    }

    if output_dir is not None:
        out_path = _as_path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out_path / "holdout_eval.csv", index=False)

    return {
        "summary": summary,
        "table": results_df,
        "config": cfg,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate holdout station predictions")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--data-dir", default="dataset_scremato")
    parser.add_argument("--device", default=None)
    parser.add_argument("--markers-subdir", default="acc_test")
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--outdir", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    evaluation = evaluate_holdout_stations(
        checkpoint_path=args.checkpoint,
        green_dir=args.green_dir,
        data_dir=args.data_dir,
        device=args.device,
        markers_subdir=args.markers_subdir,
        days=args.days,
        max_samples=args.max_samples,
        output_dir=args.outdir,
    )
    for key, value in evaluation["summary"].items():
        print(f"{key}: {value}", flush=True)


if __name__ == "__main__":
    main()
