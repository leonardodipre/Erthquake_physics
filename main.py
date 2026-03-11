from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from config import DEFAULT_CONFIG
from dataset import PINNDataloader
from fault import load_fault_coordinates
from loss import PINNLoss
from model import HybridPINN, load_green_matrices


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


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
    mask: torch.Tensor,
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
        f"mask={mask.device} | "
        f"u_surface={out_data['u_surface'].device} | "
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
    )

    pinn = _build_model(cfg=cfg, K_cd=K_cd, K_ij=K_ij, n_patches=n_patches, device=device)

    losses = PINNLoss(neighbors=neighbors, sigma_n=pinn.sigma_n)
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
    checkpoint = _as_path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    total_steps = int(cfg["total_steps"])
    log_every = max(1, int(cfg["log_every"]))

    print(
        f"Training start | device={device} | steps={total_steps} | "
        f"log_every={log_every} | checkpoint={checkpoint}",
        flush=True,
    )

    for step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)

        data_samples = dataloader.sample_data_batch(batch_size=int(cfg["data_batch_size"]))
        L_data = torch.zeros((), dtype=torch.float32, device=device)
        out_data: dict[str, torch.Tensor] | None = None
        batch_u_obs_device: torch.Tensor | None = None
        batch_mask_device: torch.Tensor | None = None
        for sample in data_samples:
            u_obs_device = sample["u_observed"].to(device=device, non_blocking=True)
            mask_device = sample["mask_data"].to(device=device, non_blocking=True)
            out_data = pinn(xi, eta, sample["t_seconds"])
            L_data = L_data + losses.data(
                u_pred=out_data["u_surface"],
                u_obs=u_obs_device,
                mask=mask_device,
            )
            batch_u_obs_device = u_obs_device
            batch_mask_device = mask_device
        L_data = L_data / max(len(data_samples), 1)

        if step == 0 and out_data is not None and batch_u_obs_device is not None and batch_mask_device is not None:
            _log_device_state(
                model=pinn,
                xi=xi,
                eta=eta,
                u_obs=batch_u_obs_device,
                mask=batch_mask_device,
                out_data=out_data,
            )

        collocation_times = dataloader.sample_collocation_batch(batch_size=int(cfg["n_colloc_per_step"]))
        L_rsf = torch.zeros((), dtype=torch.float32, device=device)
        L_state = torch.zeros((), dtype=torch.float32, device=device)
        L_smooth = torch.zeros((), dtype=torch.float32, device=device)
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
        n_colloc = max(len(collocation_times), 1)
        L_rsf = L_rsf / n_colloc
        L_state = L_state / n_colloc
        L_smooth = L_smooth / n_colloc

        w_rsf, w_state = PINNLoss.anneal_weights(
            step=step,
            lambda_rsf=float(cfg["lambda_rsf"]),
            lambda_state=float(cfg["lambda_state"]),
        )
        L_total = (
            float(cfg["lambda_data"]) * L_data
            + w_rsf * L_rsf
            + w_state * L_state
            + float(cfg["lambda_smooth"]) * L_smooth
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
                    "L_rsf": float(L_rsf.detach().cpu()),
                    "L_state": float(L_state.detach().cpu()),
                    "L_smooth": float(L_smooth.detach().cpu()),
                    "a_minus_b_min": float(ab.min().detach().cpu()),
                    "a_minus_b_max": float(ab.max().detach().cpu()),
                    "tau_0_mean": float(pinn.tau_0.mean().detach().cpu()),
                    "tau_dot_mean": float(pinn.tau_dot.mean().detach().cpu()),
                }
                history.append(log_entry)
                print(
                    f"Step {step + 1:5d}/{total_steps:5d} | "
                    f"L_tot={log_entry['L_total']:.2e} | "
                    f"L_data={log_entry['L_data']:.2e} | "
                    f"L_rsf={log_entry['L_rsf']:.2e} | "
                    f"L_state={log_entry['L_state']:.2e} | "
                    f"L_smooth={log_entry['L_smooth']:.2e} | "
                    f"(a-b)=[{log_entry['a_minus_b_min']:.4f},{log_entry['a_minus_b_max']:.4f}] | "
                    f"tau_0_mean={log_entry['tau_0_mean']:.2e} | "
                    f"tau_dot_mean={log_entry['tau_dot_mean']:.2e}"
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
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg: dict[str, Any] = {}
    if args.steps is not None:
        cfg["total_steps"] = args.steps
    if args.log_every is not None:
        cfg["log_every"] = max(1, int(args.log_every))

    train_pinn(
        config=cfg,
        green_dir=args.green_dir,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
