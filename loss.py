from __future__ import annotations

import torch
from torch import Tensor


def compute_L_data(u_pred: Tensor, u_obs: Tensor, mask: Tensor) -> Tensor:
    residual = (u_pred - u_obs) * mask
    n_valid = mask.sum().clamp(min=1.0)
    return torch.sum(residual.square()) / n_valid


def compute_L_rsf(tau_elastic: Tensor, tau_rsf: Tensor, sigma_n: float) -> Tensor:
    return torch.mean(((tau_elastic - tau_rsf) / sigma_n).square())


def compute_L_state(dtheta_dt: Tensor, V: Tensor, theta: Tensor, d_c: Tensor) -> Tensor:
    aging_target = 1.0 - V * theta / d_c
    return torch.mean((dtheta_dt - aging_target).square())


def compute_L_smooth(slip: Tensor, neighbors: list[list[int]]) -> Tensor:
    s_flat = slip.squeeze(-1)
    diffs = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if j > i:
                diffs.append((s_flat[i] - s_flat[j]).square())
    if not diffs:
        return torch.zeros((), dtype=slip.dtype, device=slip.device)
    return torch.mean(torch.stack(diffs))


class PINNLoss:
    def __init__(self, neighbors: list[list[int]], sigma_n: float) -> None:
        self.neighbors = neighbors
        self.sigma_n = sigma_n
        edges = []
        for i, nbrs in enumerate(neighbors):
            for j in nbrs:
                if j > i:
                    edges.append((i, j))
        if edges:
            self._edge_index_cpu = torch.tensor(edges, dtype=torch.long)
        else:
            self._edge_index_cpu = torch.empty((0, 2), dtype=torch.long)
        self._edge_index_cache: dict[str, Tensor] = {}

    def data(self, u_pred: Tensor, u_obs: Tensor, mask: Tensor) -> Tensor:
        return compute_L_data(u_pred=u_pred, u_obs=u_obs, mask=mask)

    def rsf(self, tau_elastic: Tensor, tau_rsf: Tensor) -> Tensor:
        return compute_L_rsf(tau_elastic=tau_elastic, tau_rsf=tau_rsf, sigma_n=self.sigma_n)

    def state(self, dtheta_dt: Tensor, V: Tensor, theta: Tensor, d_c: Tensor) -> Tensor:
        return compute_L_state(dtheta_dt=dtheta_dt, V=V, theta=theta, d_c=d_c)

    def smooth(self, slip: Tensor) -> Tensor:
        if self._edge_index_cpu.numel() == 0:
            return torch.zeros((), dtype=slip.dtype, device=slip.device)

        cache_key = str(slip.device)
        edge_index = self._edge_index_cache.get(cache_key)
        if edge_index is None:
            edge_index = self._edge_index_cpu.to(device=slip.device, non_blocking=True)
            self._edge_index_cache[cache_key] = edge_index

        s_flat = slip.squeeze(-1)
        diffs = s_flat[edge_index[:, 0]] - s_flat[edge_index[:, 1]]
        return torch.mean(diffs.square())

    @staticmethod
    def anneal_weights(
        step: int,
        lambda_rsf: float,
        lambda_state: float,
        warmup_start: int = 5_000,
        warmup_end: int = 15_000,
    ) -> tuple[float, float]:
        if step < warmup_start:
            return 0.0, 0.0
        if step < warmup_end:
            progress = (step - warmup_start) / max(warmup_end - warmup_start, 1)
            return lambda_rsf * progress, lambda_state * progress
        return lambda_rsf, lambda_state
