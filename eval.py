from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from dataset import PINNDataloader
from predict import load_trained_model


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
    )

    data_indices = dataloader.data_indices
    if max_samples is not None:
        data_indices = data_indices[: int(max_samples)]

    rows: list[dict[str, Any]] = []
    weighted_sse = 0.0
    weighted_abs = 0.0
    total_valid = 0.0

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

        with torch.enable_grad():
            out = model(xi, eta, t_seconds)

        residual = ((out["u_surface"] - u_obs) * mask).detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()
        metrics = _masked_metrics(residual=residual, mask=mask_np)

        valid = metrics["n_valid_components"]
        weighted_sse += (metrics["rmse_m"] ** 2) * valid
        weighted_abs += metrics["mae_m"] * valid
        total_valid += valid

        rows.append(
            {
                "index": int(row_idx),
                "date": timestamp.isoformat(),
                "time_seconds": t_seconds,
                "n_valid_components": int(valid),
                "n_valid_stations": int(valid // 3),
                "rmse_m": metrics["rmse_m"],
                "rmse_mm": metrics["rmse_m"] * 1e3,
                "mae_m": metrics["mae_m"],
                "mae_mm": metrics["mae_m"] * 1e3,
                "max_abs_m": metrics["max_abs_m"],
                "max_abs_mm": metrics["max_abs_m"] * 1e3,
            }
        )

        if eval_idx == 1 or eval_idx % 250 == 0 or eval_idx == len(data_indices):
            print(
                f"Eval {eval_idx:5d}/{len(data_indices):5d} | "
                f"date={timestamp.date()} | "
                f"rmse_mm={metrics['rmse_m'] * 1e3:.3f} | "
                f"mae_mm={metrics['mae_m'] * 1e3:.3f} | "
                f"valid_comp={int(valid)}",
                flush=True,
            )

    results_df = pd.DataFrame(rows)
    global_rmse_m = float(np.sqrt(weighted_sse / max(total_valid, 1.0)))
    global_mae_m = float(weighted_abs / max(total_valid, 1.0))

    summary = {
        "n_time_samples": int(len(results_df)),
        "time_range": (
            results_df["date"].iloc[0] if not results_df.empty else None,
            results_df["date"].iloc[-1] if not results_df.empty else None,
        ),
        "total_valid_components": int(total_valid),
        "global_rmse_m": global_rmse_m,
        "global_rmse_mm": global_rmse_m * 1e3,
        "global_mae_m": global_mae_m,
        "global_mae_mm": global_mae_m * 1e3,
        "median_time_rmse_mm": float(results_df["rmse_mm"].median()) if not results_df.empty else 0.0,
        "p90_time_rmse_mm": float(results_df["rmse_mm"].quantile(0.90)) if not results_df.empty else 0.0,
        "worst_time_rmse_mm": float(results_df["rmse_mm"].max()) if not results_df.empty else 0.0,
        "mean_valid_stations": float(results_df["n_valid_stations"].mean()) if not results_df.empty else 0.0,
        "median_valid_stations": float(results_df["n_valid_stations"].median()) if not results_df.empty else 0.0,
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
