from __future__ import annotations

import torch
from torch import Tensor


def _smooth_l1_per_element(pred: Tensor, target: Tensor, beta: float) -> Tensor:
    if beta <= 0.0:
        return (pred - target).abs()

    abs_diff = (pred - target).abs()
    quadratic = torch.minimum(abs_diff, torch.full_like(abs_diff, beta))
    linear = abs_diff - quadratic
    return 0.5 * quadratic.square() / beta + linear


def _build_component_weights(
    pred: Tensor,
    component_weights: tuple[float, float, float],
) -> Tensor:
    weights = torch.tensor(component_weights, dtype=pred.dtype, device=pred.device)
    if weights.shape != (3,):
        raise ValueError("component_weights must contain exactly 3 values.")
    if torch.any(weights <= 0.0):
        raise ValueError("component_weights must be strictly positive.")

    normalized = weights / weights.mean()
    n_components = pred.shape[-1]
    if n_components % 3 != 0:
        raise ValueError("The last dimension must be a multiple of 3.")
    repeats = n_components // 3
    return normalized.repeat(repeats)


def _compute_masked_smooth_l1(
    pred: Tensor,
    target: Tensor,
    mask: Tensor,
    beta: float,
    component_weights: tuple[float, float, float],
) -> Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must share the same shape.")

    weights = _build_component_weights(pred=pred, component_weights=component_weights)
    while weights.ndim < pred.ndim:
        weights = weights.unsqueeze(0)

    loss = _smooth_l1_per_element(pred=pred, target=target, beta=beta)
    weighted_mask = mask * weights
    normalizer = weighted_mask.sum().clamp(min=1.0)
    return torch.sum(loss * weighted_mask) / normalizer


def compute_L_data(
    u_pred: Tensor,
    u_obs: Tensor,
    mask: Tensor,
    beta: float = 5e-3,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    return _compute_masked_smooth_l1(
        pred=u_pred,
        target=u_obs,
        mask=mask,
        beta=beta,
        component_weights=component_weights,
    )


def compute_L_velocity(
    v_pred: Tensor,
    v_obs: Tensor,
    mask: Tensor,
    scale: float = 1e-10,
    beta: float = 1.0,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Tensor:
    scaled_pred = v_pred / scale
    scaled_obs = v_obs / scale
    return _compute_masked_smooth_l1(
        pred=scaled_pred,
        target=scaled_obs,
        mask=mask,
        beta=beta,
        component_weights=component_weights,
    )


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
    def __init__(
        self,
        neighbors: list[list[int]],
        sigma_n: float,
        surface_velocity_scale: float = 1e-10,
        position_huber_beta: float = 5e-3,
        velocity_huber_beta: float = 1.0,
        position_component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
        velocity_component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        self.neighbors = neighbors
        self.sigma_n = sigma_n
        self.surface_velocity_scale = float(surface_velocity_scale)
        self.position_huber_beta = float(position_huber_beta)
        self.velocity_huber_beta = float(velocity_huber_beta)
        self.position_component_weights = tuple(float(v) for v in position_component_weights)
        self.velocity_component_weights = tuple(float(v) for v in velocity_component_weights)
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
        return compute_L_data(
            u_pred=u_pred,
            u_obs=u_obs,
            mask=mask,
            beta=self.position_huber_beta,
            component_weights=self.position_component_weights,
        )

    def velocity(self, v_pred: Tensor, v_obs: Tensor, mask: Tensor) -> Tensor:
        return compute_L_velocity(
            v_pred=v_pred,
            v_obs=v_obs,
            mask=mask,
            scale=self.surface_velocity_scale,
            beta=self.velocity_huber_beta,
            component_weights=self.velocity_component_weights,
        )

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
