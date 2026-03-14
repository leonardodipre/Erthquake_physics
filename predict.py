from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from config import DEFAULT_CONFIG
from fault import load_fault_coordinates
from model import HybridPINN, load_green_matrices


REPO_ROOT = Path(__file__).resolve().parent


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _coerce_time_seconds(
    time_seconds: float | None,
    time_date: str | None,
    reference_date: str,
) -> float:
    if (time_seconds is None) == (time_date is None):
        raise ValueError("Specify exactly one between --time-seconds and --time-date.")
    if time_seconds is not None:
        return float(time_seconds)
    return float((pd.Timestamp(time_date) - pd.Timestamp(reference_date)).total_seconds())


def _build_model(
    cfg: dict[str, Any],
    K_cd: torch.Tensor,
    K_ij: torch.Tensor,
    n_patches: int,
    device: torch.device,
) -> HybridPINN:
    model = HybridPINN(
        K_cd=K_cd,
        K_ij=K_ij,
        n_patches=n_patches,
        mu_0=float(cfg["mu_0"]),
        V_0=float(cfg["V_0"]),
        sigma_n=float(cfg["sigma_n"]),
        slip_hidden=int(cfg["slip_hidden"]),
        slip_blocks=int(cfg["slip_blocks"]),
        friction_hidden=int(cfg["friction_hidden"]),
        friction_blocks=int(cfg["friction_blocks"]),
        dropout=float(cfg["dropout"]),
        tau_dot_init=float(cfg.get("tau_dot_init", 0.1)),
        time_input_scale=float(cfg.get("time_input_scale", 1e8)),
    )
    return model.to(device)


def _resolve_green_dir(green_dir: str | Path, checkpoint_path: str | Path) -> Path:
    green_dir_path = _as_path(green_dir)
    if green_dir_path.is_absolute():
        return green_dir_path

    checkpoint_dir = _as_path(checkpoint_path).resolve().parent
    candidates = (
        checkpoint_dir / green_dir_path,
        REPO_ROOT / green_dir_path,
        Path.cwd() / green_dir_path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return (REPO_ROOT / green_dir_path).resolve()


def load_trained_model(
    checkpoint_path: str,
    green_dir: str = "green_out",
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    device_t = _resolve_device(device)
    payload = torch.load(_as_path(checkpoint_path), map_location=device_t)

    saved_cfg: dict[str, Any] = {}
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        saved_cfg.update(payload.get("config", {}))
        green_dir = str(payload.get("green_dir", green_dir))
    else:
        state_dict = payload

    green_dir_path = _resolve_green_dir(green_dir=green_dir, checkpoint_path=checkpoint_path)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update(saved_cfg)

    K_cd, K_ij, summary = load_green_matrices(green_dir=green_dir_path, device=device_t)
    n_patches = int(summary["Nc"])
    xi, eta, _ = load_fault_coordinates(green_dir=green_dir_path)
    xi = xi.to(device_t)
    eta = eta.to(device_t)

    model = _build_model(cfg=cfg, K_cd=K_cd, K_ij=K_ij, n_patches=n_patches, device=device_t)
    model.load_state_dict(state_dict)
    model.eval()

    return {
        "model": model,
        "config": cfg,
        "green_dir": str(green_dir_path),
        "summary": summary,
        "xi": xi,
        "eta": eta,
        "device": device_t,
    }


def predict_snapshot(
    checkpoint_path: str,
    green_dir: str = "green_out",
    device: torch.device | str | None = None,
    time_seconds: float | None = None,
    time_date: str | None = None,
    output_path: str | None = None,
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

    t_seconds = _coerce_time_seconds(
        time_seconds=time_seconds,
        time_date=time_date,
        reference_date=str(cfg["reference_date"]),
    )

    with torch.enable_grad():
        out = model(xi, eta, t_seconds)

    arrays = {key: value.detach().cpu().numpy() for key, value in out.items()}
    arrays["a_minus_b"] = arrays["a"] - arrays["b"]
    summary = {
        "time_seconds": float(t_seconds),
        "u_surface_abs_max_m": float(np.abs(arrays["u_surface"]).max()),
        "slip_range_m": (float(arrays["s"].min()), float(arrays["s"].max())),
        "V_abs_max_m_per_s": float(np.abs(arrays["V"]).max()),
        "theta_range_s": (float(arrays["theta"].min()), float(arrays["theta"].max())),
        "a_minus_b_range": (float(arrays["a_minus_b"].min()), float(arrays["a_minus_b"].max())),
        "D_c_range_m": (float(arrays["D_c"].min()), float(arrays["D_c"].max())),
        "tau_elastic_range_pa": (
            float(arrays["tau_elastic"].min()),
            float(arrays["tau_elastic"].max()),
        ),
        "tau_rsf_range_pa": (
            float(arrays["tau_rsf"].min()),
            float(arrays["tau_rsf"].max()),
        ),
        "v_surface_abs_max_m_per_s": float(np.abs(arrays["v_surface"]).max()),
        "v_surface_abs_max_mm_per_day": float(np.abs(arrays["v_surface"]).max()) * 1e3 * 86400.0,
    }

    if output_path is not None:
        out_path = _as_path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out_path, **arrays, time_seconds=np.array([t_seconds], dtype=np.float64))

    return {
        "summary": summary,
        "arrays": arrays,
        "config": cfg,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run prediction from a trained model")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--device", default=None)
    parser.add_argument("--time-seconds", type=float, default=None)
    parser.add_argument("--time-date", default=None)
    parser.add_argument("--out", default=None)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    pred = predict_snapshot(
        checkpoint_path=args.checkpoint,
        green_dir=args.green_dir,
        device=args.device,
        time_seconds=args.time_seconds,
        time_date=args.time_date,
        output_path=args.out,
    )
    for key, value in pred["summary"].items():
        print(f"{key}: {value}", flush=True)
    if args.out is not None:
        print(f"prediction_saved: {args.out}", flush=True)


if __name__ == "__main__":
    main()
