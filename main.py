from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from config import DEFAULT_CONFIG
from dataset import PINNDataloader, compute_station_geo_weights, list_markers_in_subdir
from fault import load_fault_coordinates
from loss import PINNLoss
from model import HybridPINN, load_green_matrices


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_excluded_markers(cfg: Mapping[str, Any], data_dir: str) -> list[str]:
    explicit = [str(marker).strip().upper() for marker in cfg.get("excluded_markers", ()) if str(marker).strip()]
    by_subdir = []
    subdir = cfg.get("excluded_markers_subdir")
    if subdir:
        by_subdir = list_markers_in_subdir(data_dir=data_dir, subdir=str(subdir))
    return sorted(dict.fromkeys(explicit + by_subdir))


def _prepare_runtime(
    green_dir: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any], int, int, torch.Tensor, torch.Tensor, list[list[int]]]:
    K_cd, K_ij, summary = load_green_matrices(green_dir=green_dir, device=device)
    n_obs = len(np.load(_as_path(green_dir) / "station_ids.npy", allow_pickle=True)) * 3
    n_patches = int(summary["Nc"])

    assert K_cd.shape == (n_obs, n_patches), f"Unexpected K_cd shape: {tuple(K_cd.shape)}"
    assert K_ij.shape == (n_patches, n_patches), f"Unexpected K_ij shape: {tuple(K_ij.shape)}"

    xi, eta, neighbors = load_fault_coordinates(green_dir=green_dir)
    return K_cd, K_ij, summary, n_obs, n_patches, xi.to(device), eta.to(device), neighbors


def _build_model(
    cfg: Mapping[str, Any],
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


def _save_checkpoint(
    checkpoint_path: Path,
    model: HybridPINN,
    config: Mapping[str, Any],
    green_dir: str,
    data_dir: str,
    summary: Mapping[str, Any],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "config": dict(config),
        "green_dir": str(green_dir),
        "data_dir": str(data_dir),
        "summary": dict(summary),
    }
    torch.save(payload, checkpoint_path)


def _log_device_state(
    model: HybridPINN,
    xi: torch.Tensor,
    eta: torch.Tensor,
    u_obs: torch.Tensor,
    v_obs: torch.Tensor,
    mask: torch.Tensor,
    velocity_mask: torch.Tensor,
    out_data: dict[str, torch.Tensor],
) -> None:
    first_param = next(model.parameters())
    print(
        "Device check | "
        f"param={first_param.device} | "
        f"K_cd={model.K_cd.device} | "
        f"K_ij={model.K_ij.device} | "
        f"xi={xi.device} | "
        f"eta={eta.device} | "
        f"u_obs={u_obs.device} | "
        f"v_obs={v_obs.device} | "
        f"mask={mask.device} | "
        f"mask_v={velocity_mask.device} | "
        f"u_surface={out_data['u_surface'].device} | "
        f"v_surface={out_data['v_surface'].device} | "
        f"s={out_data['s'].device}",
        flush=True,
    )


def train_pinn(
    config: Mapping[str, Any] | None = None,
    green_dir: str = "green_out",
    data_dir: str = "dataset_scremato",
    checkpoint_path: str = "checkpoints/model.pt",
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config is not None:
        cfg.update(dict(config))

    device = _resolve_device(device)
    excluded_markers = _resolve_excluded_markers(cfg=cfg, data_dir=data_dir)
    cfg["excluded_markers"] = list(excluded_markers)
    print("We are on device", device)
    if cfg.get("seed") is not None:
        np.random.seed(int(cfg["seed"]))
        torch.manual_seed(int(cfg["seed"]))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(cfg["seed"]))

    K_cd, K_ij, summary, _, n_patches, xi, eta, neighbors = _prepare_runtime(
        green_dir=green_dir,
        device=device,
    )

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
        excluded_markers=excluded_markers,
        max_station_jump_m=cfg.get("max_station_jump_m"),
        savgol_window=int(cfg.get("savgol_window", 0)),
        savgol_polyorder=int(cfg.get("savgol_polyorder", 3)),
    )

    checkpoint = _as_path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    pinn = _build_model(cfg=cfg, K_cd=K_cd, K_ij=K_ij, n_patches=n_patches, device=device)

    # Geometric station weighting
    station_weights_tensor: torch.Tensor | None = None
    if cfg.get("geo_weight_enabled", False):
        import json as _json_mod
        stations_json = str(cfg.get("stations_json", "dataset_scremato/stations_ITA_laquila_150km.json"))
        station_ids_path = str(_as_path(green_dir) / "station_ids.npy")
        geo_weights, geo_diag = compute_station_geo_weights(
            station_ids_path=station_ids_path,
            stations_json_path=stations_json,
            k=int(cfg.get("geo_weight_k", 5)),
            alpha=float(cfg.get("geo_weight_alpha", 0.5)),
            clip_min=float(cfg.get("geo_weight_clip_min", 0.5)),
            clip_max=float(cfg.get("geo_weight_clip_max", 2.0)),
        )
        station_weights_tensor = torch.from_numpy(geo_weights).to(device)
        # Save diagnostics
        diag_path = checkpoint.parent / "station_geo_weights.json"
        with open(diag_path, "w") as f:
            _json_mod.dump(geo_diag, f, indent=2)
        w_vals = [d["weight"] for d in geo_diag]
        print(
            f"Geo weights | min={min(w_vals):.3f} max={max(w_vals):.3f} "
            f"mean={np.mean(w_vals):.3f} | saved to {diag_path}",
            flush=True,
        )

    losses = PINNLoss(
        neighbors=neighbors,
        sigma_n=pinn.sigma_n,
        surface_velocity_scale=float(cfg.get("surface_velocity_scale", 1e-10)),
        position_huber_beta=float(cfg.get("position_huber_beta", 5e-3)),
        velocity_huber_beta=float(cfg.get("velocity_huber_beta", 1.0)),
        position_component_weights=tuple(cfg.get("position_component_weights", (1.0, 1.0, 1.0))),
        velocity_component_weights=tuple(cfg.get("velocity_component_weights", (1.0, 1.0, 1.0))),
        station_weights=station_weights_tensor,
        nx=summary.get("nx"),
        ny=summary.get("ny"),
        boundary_decay_xi=int(cfg.get("boundary_decay_xi", 5)),
        boundary_decay_eta=int(cfg.get("boundary_decay_eta", 3)),
        boundary_shallow_factor=float(cfg.get("boundary_shallow_factor", 2.0)),
        boundary_left_factor=float(cfg.get("boundary_left_factor", 2.0)),
    )
    optimizer = torch.optim.AdamW(
        pinn.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["total_steps"]),
        eta_min=float(cfg["lr_min"]),
    )

    history: list[dict[str, float]] = []
    total_steps = int(cfg["total_steps"])
    log_every = max(1, int(cfg["log_every"]))

    print(
        f"Training start | device={device} | steps={total_steps} | "
        f"log_every={log_every} | checkpoint={checkpoint}",
        flush=True,
    )
    if excluded_markers:
        print(
            f"Holdout stations excluded from train: {', '.join(excluded_markers)}",
            flush=True,
        )
    savgol_w = int(cfg.get("savgol_window", 0))
    print(
        "Dataset filter | "
        f"pos_valid={dataloader.filter_report['position_valid_fraction']:.3f} | "
        f"vel_valid={dataloader.filter_report['velocity_valid_fraction']:.3f} | "
        f"robust_excluded={dataloader.filter_report['robust_excluded_fraction']:.3f} | "
        f"savgol={'off' if savgol_w == 0 else f'{savgol_w}d'}",
        flush=True,
    )
    quality = dataloader.station_quality_report()
    top_jumpers = [q for q in quality[:10] if q["max_jump_m"] > 0.01]
    if top_jumpers:
        print("Top stations by max daily jump (m):", flush=True)
        for q in top_jumpers:
            tag = " [EXCLUDED]" if q["marker"] in set(dataloader.excluded_markers) else ""
            print(f"  {q['marker']:16s}  jump={q['max_jump_m']:.4f}  max_pos={q['max_abs_position_m']:.4f}{tag}", flush=True)

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        data_samples = dataloader.sample_data_batch(batch_size=int(cfg["data_batch_size"]))
        L_position = torch.zeros((), dtype=torch.float32, device=device)
        L_velocity = torch.zeros((), dtype=torch.float32, device=device)
        out_data: dict[str, torch.Tensor] | None = None
        batch_u_obs_device: torch.Tensor | None = None
        batch_v_obs_device: torch.Tensor | None = None
        batch_mask_device: torch.Tensor | None = None
        batch_velocity_mask_device: torch.Tensor | None = None
        for sample in data_samples:
            u_obs_device = sample["u_observed"].to(device=device, non_blocking=True)
            v_obs_device = sample["v_observed"].to(device=device, non_blocking=True)
            mask_device = sample["mask_data"].to(device=device, non_blocking=True)
            velocity_mask_device = sample["mask_velocity"].to(device=device, non_blocking=True)
            out_data = pinn(xi, eta, sample["t_seconds"])
            L_position = L_position + losses.data(
                u_pred=out_data["u_surface"],
                u_obs=u_obs_device,
                mask=mask_device,
            )
            L_velocity = L_velocity + losses.velocity(
                v_pred=out_data["v_surface"],
                v_obs=v_obs_device,
                mask=velocity_mask_device,
            )
            batch_u_obs_device = u_obs_device
            batch_v_obs_device = v_obs_device
            batch_mask_device = mask_device
            batch_velocity_mask_device = velocity_mask_device
        L_position = L_position / max(len(data_samples), 1)
        L_velocity = L_velocity / max(len(data_samples), 1)
        L_data = (
            float(cfg.get("lambda_data_position", 1.0)) * L_position
            + float(cfg.get("lambda_data_velocity", 1.0)) * L_velocity
        )

        if (
            step == 0
            and out_data is not None
            and batch_u_obs_device is not None
            and batch_v_obs_device is not None
            and batch_mask_device is not None
            and batch_velocity_mask_device is not None
        ):
            _log_device_state(
                model=pinn,
                xi=xi,
                eta=eta,
                u_obs=batch_u_obs_device,
                v_obs=batch_v_obs_device,
                mask=batch_mask_device,
                velocity_mask=batch_velocity_mask_device,
                out_data=out_data,
            )

        collocation_times = dataloader.sample_collocation_batch(batch_size=int(cfg["n_colloc_per_step"]))
        collocation_times.sort()
        L_rsf = torch.zeros((), dtype=torch.float32, device=device)
        L_state = torch.zeros((), dtype=torch.float32, device=device)
        L_smooth = torch.zeros((), dtype=torch.float32, device=device)
        L_friction_reg = torch.zeros((), dtype=torch.float32, device=device)
        L_V_temporal = torch.zeros((), dtype=torch.float32, device=device)
        L_boundary_V = torch.zeros((), dtype=torch.float32, device=device)
        prev_V: torch.Tensor | None = None
        boundary_V_ref = float(cfg.get("boundary_V_ref", 1e-9))
        for t_colloc in collocation_times:
            out_phys = pinn(xi, eta, t_colloc)
            L_rsf = L_rsf + losses.rsf(
                tau_elastic=out_phys["tau_elastic"],
                tau_rsf=out_phys["tau_rsf"],
            )
            L_state = L_state + losses.state(
                dtheta_dt=out_phys["dtheta_dt"],
                V=out_phys["V"],
                theta=out_phys["theta"],
                d_c=out_phys["D_c"],
            )
            L_smooth = L_smooth + losses.smooth(out_phys["s"])
            L_friction_reg = L_friction_reg + losses.friction_reg(
                a=out_phys["a"],
                b=out_phys["b"],
                target_ab_mean=float(cfg.get("target_ab_mean", 0.0)),
                target_ab_std=float(cfg.get("target_ab_std", 0.008)),
            )
            L_boundary_V = L_boundary_V + losses.boundary_V(
                V=out_phys["V"], V_ref=boundary_V_ref,
            )
            if prev_V is not None:
                L_V_temporal = L_V_temporal + losses.V_temporal(
                    V_current=out_phys["V"],
                    V_other=prev_V,
                )
            prev_V = out_phys["V"]
        n_colloc = max(len(collocation_times), 1)
        L_rsf = L_rsf / n_colloc
        L_state = L_state / n_colloc
        L_smooth = L_smooth / n_colloc
        L_friction_reg = L_friction_reg / n_colloc
        L_boundary_V = L_boundary_V / n_colloc
        L_V_temporal = L_V_temporal / max(n_colloc - 1, 1)

        w_rsf, w_state = PINNLoss.anneal_weights(
            step=step,
            lambda_rsf=float(cfg["lambda_rsf"]),
            lambda_state=float(cfg["lambda_state"]),
            warmup_start=int(cfg.get("warmup_start", 5_000)),
            warmup_end=int(cfg.get("warmup_end", 15_000)),
        )
        L_total = (
            float(cfg["lambda_data"]) * L_data
            + w_rsf * L_rsf
            + w_state * L_state
            + float(cfg["lambda_smooth"]) * L_smooth
            + float(cfg.get("lambda_friction_reg", 0.0)) * L_friction_reg
            + float(cfg.get("lambda_V_temporal", 0.0)) * L_V_temporal
            + float(cfg.get("lambda_boundary_V", 0.0)) * L_boundary_V
        )

        L_total.backward()
        torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=float(cfg["grad_clip"]))
        optimizer.step()
        scheduler.step()

        if step == 0 or (step + 1) % log_every == 0 or step == total_steps - 1:
            if out_data is None:
                raise RuntimeError("out_data is unexpectedly None during logging.")
            with torch.no_grad():
                ab = (out_data["a"] - out_data["b"]).squeeze(-1)
                log_entry = {
                    "step": float(step),
                    "L_total": float(L_total.detach().cpu()),
                    "L_data": float(L_data.detach().cpu()),
                    "L_position": float(L_position.detach().cpu()),
                    "L_velocity": float(L_velocity.detach().cpu()),
                    "L_rsf": float(L_rsf.detach().cpu()),
                    "L_state": float(L_state.detach().cpu()),
                    "L_smooth": float(L_smooth.detach().cpu()),
                    "L_friction_reg": float(L_friction_reg.detach().cpu()),
                    "L_V_temporal": float(L_V_temporal.detach().cpu()),
                    "L_boundary_V": float(L_boundary_V.detach().cpu()),
                    "a_minus_b_min": float(ab.min().detach().cpu()),
                    "a_minus_b_max": float(ab.max().detach().cpu()),
                    "a_minus_b_mean": float(ab.mean().detach().cpu()),
                    "a_minus_b_std": float(ab.std().detach().cpu()),
                    "tau_0_mean": float(pinn.tau_0.mean().detach().cpu()),
                    "tau_dot_mean": float(pinn.tau_dot.mean().detach().cpu()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "w_rsf": float(w_rsf),
                    "w_state": float(w_state),
                    "V_log10_mean": float(torch.log10(out_data["V"].abs() + 1e-20).mean().detach().cpu()),
                    "V_log10_std": float(torch.log10(out_data["V"].abs() + 1e-20).std().detach().cpu()),
                    "slip_max": float(out_data["s"].abs().max().detach().cpu()),
                }
                history.append(log_entry)
                print(
                    f"Step {step + 1:5d}/{total_steps:5d} | "
                    f"L_tot={log_entry['L_total']:.2e} | "
                    f"L_pos={log_entry['L_position']:.2e} | "
                    f"L_vel={log_entry['L_velocity']:.2e} | "
                    f"L_rsf={log_entry['L_rsf']:.2e} | "
                    f"L_state={log_entry['L_state']:.2e} | "
                    f"L_freg={log_entry['L_friction_reg']:.2e} | "
                    f"L_Vt={log_entry['L_V_temporal']:.2e} | "
                    f"L_bV={log_entry['L_boundary_V']:.2e} | "
                    f"(a-b)=[{log_entry['a_minus_b_min']:.4f},{log_entry['a_minus_b_max']:.4f}] "
                    f"mean={log_entry['a_minus_b_mean']:.4f} std={log_entry['a_minus_b_std']:.4f}"
                    ,
                    flush=True,
                )

    _save_checkpoint(
        checkpoint_path=checkpoint,
        model=pinn,
        config=cfg,
        green_dir=green_dir,
        data_dir=data_dir,
        summary=summary,
    )
    print(f"Training done | checkpoint saved to {checkpoint}", flush=True)

    if history:
        import pandas as pd
        history_path = checkpoint.with_suffix(".history.csv")
        pd.DataFrame(history).to_csv(history_path, index=False)
        print(f"Training history saved to {history_path}", flush=True)
    return {
        "model": pinn,
        "history": history,
        "dataloader": dataloader,
        "config": cfg,
        "checkpoint_path": checkpoint,
    }

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the PINN model")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--data-dir", default="dataset_scremato")
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--config-json",
        default=None,
        help="JSON string or path to JSON file with config overrides",
    )
    return parser


def main() -> None:
    import json as _json
    import os as _os

    parser = _build_parser()
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if args.config_json is not None:
        if _os.path.isfile(args.config_json):
            with open(args.config_json) as f:
                cfg.update(_json.load(f))
        else:
            cfg.update(_json.loads(args.config_json))
    if args.steps is not None:
        cfg["total_steps"] = args.steps
    if args.log_every is not None:
        cfg["log_every"] = max(1, int(args.log_every))
    if args.seed is not None:
        cfg["seed"] = args.seed

    train_pinn(
        config=cfg,
        green_dir=args.green_dir,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
