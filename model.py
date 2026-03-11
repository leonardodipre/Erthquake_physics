from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _scale_sigmoid(raw: Tensor, lower: float, upper: float) -> Tensor:
    return lower + (upper - lower) * torch.sigmoid(raw)


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.05) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return residual + x


class SlipNetwork(nn.Module):
    THETA_SCALE = 1e7

    def __init__(
        self,
        hidden_dim: int = 64,
        num_blocks: int = 4,
        dropout: float = 0.05,
        time_input_scale: float = 1e8,
    ) -> None:
        super().__init__()
        self.time_input_scale = float(time_input_scale)
        self.input_proj = nn.Linear(3, hidden_dim)
        self.input_act = nn.GELU()
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head_s = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.head_theta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, xi: Tensor, eta: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        scaled_t = t / self.time_input_scale
        x = torch.stack([xi, eta, scaled_t], dim=-1)
        x = self.input_proj(x)
        x = self.input_act(x)
        for block in self.blocks:
            x = block(x)
        s = self.head_s(x)
        theta = self.head_theta(x) * self.THETA_SCALE
        return s, theta


class FrictionNetwork(nn.Module):
    def __init__(self, hidden_dim: int = 32, num_blocks: int = 3, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2, hidden_dim)
        self.input_act = nn.GELU()
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, xi: Tensor, eta: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x = torch.stack([xi, eta], dim=-1)
        x = self.input_proj(x)
        x = self.input_act(x)
        for block in self.blocks:
            x = block(x)
        raw = self.head(x)
        a = _scale_sigmoid(raw[:, 0:1], 0.001, 0.050)
        b = _scale_sigmoid(raw[:, 1:2], 0.001, 0.050)
        d_c = _scale_sigmoid(raw[:, 2:3], 1e-4, 0.050)
        return a, b, d_c


class HybridPINN(nn.Module):
    def __init__(
        self,
        K_cd: Tensor,
        K_ij: Tensor,
        n_patches: int,
        mu_0: float = 0.6,
        V_0: float = 1e-9,
        sigma_n: float = 100e6,
        slip_hidden: int = 64,
        slip_blocks: int = 4,
        friction_hidden: int = 32,
        friction_blocks: int = 3,
        dropout: float = 0.05,
        tau_dot_init: float = 0.1,
        time_input_scale: float = 1e8,
    ) -> None:
        super().__init__()
        self.f_slip = SlipNetwork(
            hidden_dim=slip_hidden,
            num_blocks=slip_blocks,
            dropout=dropout,
            time_input_scale=time_input_scale,
        )
        self.nn_f = FrictionNetwork(
            hidden_dim=friction_hidden,
            num_blocks=friction_blocks,
            dropout=dropout,
        )

        self.register_buffer("K_cd", K_cd)
        self.register_buffer("K_ij", K_ij)

        self.mu_0 = float(mu_0)
        self.V_0 = float(V_0)
        self.sigma_n = float(sigma_n)
        self.n_patches = int(n_patches)

        self.tau_0 = nn.Parameter(
            torch.full((n_patches,), self.sigma_n * self.mu_0, dtype=torch.float32)
        )
        self.tau_dot = nn.Parameter(
            torch.full((n_patches,), float(tau_dot_init), dtype=torch.float32)
        )

    def forward(self, fault_xi: Tensor, fault_eta: Tensor, t_value: float | Tensor) -> dict[str, Tensor]:
        n_patches = fault_xi.shape[0]
        device = fault_xi.device
        dtype = fault_xi.dtype

        t_scalar = torch.as_tensor(t_value, device=device, dtype=dtype)
        t_per_patch = torch.full(
            (n_patches,),
            fill_value=float(t_scalar.detach().item()),
            device=device,
            dtype=dtype,
        )
        t_per_patch.requires_grad_(True)

        slip, theta = self.f_slip(fault_xi, fault_eta, t_per_patch)

        V = torch.autograd.grad(
            slip.sum(),
            t_per_patch,
            create_graph=True,
            retain_graph=True,
        )[0].unsqueeze(-1)
        dtheta_dt = torch.autograd.grad(
            theta.sum(),
            t_per_patch,
            create_graph=True,
            retain_graph=True,
        )[0].unsqueeze(-1)

        u_surface = self.K_cd @ slip.squeeze(-1)
        tau_interaction = self.K_ij @ slip.squeeze(-1)
        tau_elastic = (
            self.tau_0.to(device=device, dtype=dtype)
            + self.tau_dot.to(device=device, dtype=dtype) * t_per_patch
            - tau_interaction
        ).clamp(-1e10, 1e10).unsqueeze(-1)

        a, b, d_c = self.nn_f(fault_xi, fault_eta)

        eps = torch.tensor(1e-20, device=device, dtype=dtype)
        V_safe = V.abs() + eps
        theta_safe = theta.abs() + eps
        d_c_safe = d_c.clamp(min=1e-12)

        log_V = torch.clamp(torch.log(V_safe / self.V_0), -50.0, 50.0)
        log_theta = torch.clamp(torch.log(theta_safe * self.V_0 / d_c_safe), -50.0, 50.0)
        tau_rsf = self.sigma_n * (self.mu_0 + a * log_V + b * log_theta)

        return {
            "u_surface": u_surface,
            "s": slip,
            "V": V,
            "theta": theta,
            "dtheta_dt": dtheta_dt,
            "tau_elastic": tau_elastic,
            "tau_rsf": tau_rsf,
            "a": a,
            "b": b,
            "D_c": d_c,
            "tau_0": self.tau_0,
            "tau_dot": self.tau_dot,
        }


def load_green_matrices(
    green_dir: str,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    green_dir_path = _as_path(green_dir)
    summary = json.loads((green_dir_path / "green_summary.json").read_text())
    if device is None:
        device = "cpu"
    K_cd = torch.tensor(np.load(green_dir_path / "K_cd_disp.npy"), dtype=torch.float32, device=device)
    K_ij = torch.tensor(np.load(green_dir_path / "K_ij_tau.npy"), dtype=torch.float32, device=device)
    return K_cd, K_ij, summary
