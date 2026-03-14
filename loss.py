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
    station_weights: Tensor | None = None,
) -> Tensor:
    if pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("pred, target and mask must share the same shape.")

    weights = _build_component_weights(pred=pred, component_weights=component_weights)
    while weights.ndim < pred.ndim:
        weights = weights.unsqueeze(0)

    loss = _smooth_l1_per_element(pred=pred, target=target, beta=beta)
    weighted_mask = mask * weights
    if station_weights is not None:
        sw = station_weights
        while sw.ndim < pred.ndim:
            sw = sw.unsqueeze(0)
        weighted_mask = weighted_mask * sw
    normalizer = weighted_mask.sum().clamp(min=1.0)
    return torch.sum(loss * weighted_mask) / normalizer


def compute_L_data(
    u_pred: Tensor,
    u_obs: Tensor,
    mask: Tensor,
    beta: float = 5e-3,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    station_weights: Tensor | None = None,
) -> Tensor:
    return _compute_masked_smooth_l1(
        pred=u_pred,
        target=u_obs,
        mask=mask,
        beta=beta,
        component_weights=component_weights,
        station_weights=station_weights,
    )


def compute_L_velocity(
    v_pred: Tensor,
    v_obs: Tensor,
    mask: Tensor,
    scale: float = 1e-10,
    beta: float = 1.0,
    component_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    station_weights: Tensor | None = None,
) -> Tensor:
    scaled_pred = v_pred / scale
    scaled_obs = v_obs / scale
    return _compute_masked_smooth_l1(
        pred=scaled_pred,
        target=scaled_obs,
        mask=mask,
        beta=beta,
        component_weights=component_weights,
        station_weights=station_weights,
    )


def compute_L_rsf(tau_elastic: Tensor, tau_rsf: Tensor, sigma_n: float) -> Tensor:
    return torch.mean(((tau_elastic - tau_rsf) / sigma_n).square())


def compute_L_state(dtheta_dt: Tensor, V: Tensor, theta: Tensor, d_c: Tensor) -> Tensor:
    aging_target = 1.0 - V * theta / d_c
    return torch.mean((dtheta_dt - aging_target).square())


def compute_L_friction_reg(
    a: Tensor,
    b: Tensor,
    neighbors: list[list[int]] | None = None,
    edge_index: Tensor | None = None,
    target_ab_mean: float = 0.0,
    target_ab_std: float = 0.008,
) -> Tensor:
    """Regularize friction parameters to prevent (a-b) collapse.

    Two terms:
    1. Penalize deviation of mean(a-b) from target_ab_mean (prevents global
       drift toward all-negative).
    2. Encourage spatial variance: penalize if std(a-b) is too small
       (prevents uniform collapse).
    3. Spatial smoothness on (a-b) using neighbor edges (prevents salt-and-pepper).
    """
    ab = (a - b).squeeze(-1)

    # Term 1: mean(a-b) should be near target_ab_mean
    mean_penalty = (ab.mean() - target_ab_mean).square()

    # Term 2: std(a-b) should be at least target_ab_std
    ab_std = ab.std()
    std_penalty = torch.nn.functional.relu(target_ab_std - ab_std).square()

    # Term 3: spatial smoothness on (a-b) via neighbor edges
    smooth_penalty = torch.zeros((), dtype=ab.dtype, device=ab.device)
    if edge_index is not None and edge_index.numel() > 0:
        diffs = ab[edge_index[:, 0]] - ab[edge_index[:, 1]]
        smooth_penalty = torch.mean(diffs.square())

    return mean_penalty + std_penalty + 0.1 * smooth_penalty


def compute_L_V_temporal(V_current: Tensor, V_other: Tensor) -> Tensor:
    """Penalize large temporal jumps in slip rate between two collocation times.

    Computed on log-scale to handle the wide dynamic range of V.
    """
    eps = 1e-20
    log_V_curr = torch.log(V_current.abs() + eps)
    log_V_other = torch.log(V_other.abs() + eps)
    return torch.mean((log_V_curr - log_V_other).square())


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
        station_weights: Tensor | None = None,
    ) -> None:
        self.neighbors = neighbors
        self.sigma_n = sigma_n
        self.surface_velocity_scale = float(surface_velocity_scale)
        self.position_huber_beta = float(position_huber_beta)
        self.velocity_huber_beta = float(velocity_huber_beta)
        self.position_component_weights = tuple(float(v) for v in position_component_weights)
        self.velocity_component_weights = tuple(float(v) for v in velocity_component_weights)
        self.station_weights = station_weights  # shape (n_stations * 3,) or None
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
            station_weights=self.station_weights,
        )

    def velocity(self, v_pred: Tensor, v_obs: Tensor, mask: Tensor) -> Tensor:
        return compute_L_velocity(
            v_pred=v_pred,
            v_obs=v_obs,
            mask=mask,
            scale=self.surface_velocity_scale,
            beta=self.velocity_huber_beta,
            component_weights=self.velocity_component_weights,
            station_weights=self.station_weights,
        )

    def rsf(self, tau_elastic: Tensor, tau_rsf: Tensor) -> Tensor:
        return compute_L_rsf(tau_elastic=tau_elastic, tau_rsf=tau_rsf, sigma_n=self.sigma_n)

    def state(self, dtheta_dt: Tensor, V: Tensor, theta: Tensor, d_c: Tensor) -> Tensor:
        return compute_L_state(dtheta_dt=dtheta_dt, V=V, theta=theta, d_c=d_c)

    def friction_reg(
        self,
        a: Tensor,
        b: Tensor,
        target_ab_mean: float = 0.0,
        target_ab_std: float = 0.008,
    ) -> Tensor:
        cache_key = str(a.device)
        edge_index = self._edge_index_cache.get(cache_key)
        if edge_index is None and self._edge_index_cpu.numel() > 0:
            edge_index = self._edge_index_cpu.to(device=a.device, non_blocking=True)
            self._edge_index_cache[cache_key] = edge_index
        return compute_L_friction_reg(
            a=a,
            b=b,
            edge_index=edge_index,
            target_ab_mean=target_ab_mean,
            target_ab_std=target_ab_std,
        )

    @staticmethod
    def V_temporal(V_current: Tensor, V_other: Tensor) -> Tensor:
        return compute_L_V_temporal(V_current=V_current, V_other=V_other)

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
