from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial import KDTree
from torch import Tensor


def _as_path(path_like: str | Path) -> Path:
    return path_like if isinstance(path_like, Path) else Path(path_like)


def _parse_mesh_meta(meta_value: Any) -> dict[str, Any]:
    if isinstance(meta_value, np.ndarray):
        meta_value = meta_value.item()
    if isinstance(meta_value, str):
        return json.loads(meta_value)
    if isinstance(meta_value, dict):
        return meta_value
    raise TypeError(f"Unsupported mesh metadata type: {type(meta_value)!r}")


def _looks_row_major_regular(centers: np.ndarray, nx: int, ny: int, tol: float = 0.05) -> bool:
    if nx * ny != centers.shape[0]:
        return False
    horiz = []
    vert = []
    for idx in range(nx * ny):
        iy = idx // nx
        ix = idx % nx
        if ix < nx - 1:
            horiz.append(np.linalg.norm(centers[idx + 1] - centers[idx]))
        if iy < ny - 1:
            vert.append(np.linalg.norm(centers[idx + nx] - centers[idx]))
    if not horiz or not vert:
        return False
    horiz = np.asarray(horiz)
    vert = np.asarray(vert)
    horiz_rel = np.std(horiz) / max(np.median(horiz), 1e-12)
    vert_rel = np.std(vert) / max(np.median(vert), 1e-12)
    return bool(horiz_rel < tol and vert_rel < tol)


def build_neighbor_list(
    centers: np.ndarray,
    nx: int | None = None,
    ny: int | None = None,
    max_dist_factor: float = 1.5,
) -> list[list[int]]:
    n_patches = centers.shape[0]

    if (
        nx is not None
        and ny is not None
        and nx * ny == n_patches
        and _looks_row_major_regular(centers, nx=nx, ny=ny)
    ):
        neighbors: list[list[int]] = []
        for idx in range(n_patches):
            iy = idx // nx
            ix = idx % nx
            nbrs: list[int] = []
            if ix > 0:
                nbrs.append(idx - 1)
            if ix < nx - 1:
                nbrs.append(idx + 1)
            if iy > 0:
                nbrs.append(idx - nx)
            if iy < ny - 1:
                nbrs.append(idx + nx)
            neighbors.append(nbrs)
        return neighbors

    tree = KDTree(centers)
    dists, _ = tree.query(centers, k=min(2, n_patches))
    typical_dist = float(np.median(dists[:, 1])) if n_patches > 1 else 0.0
    threshold = typical_dist * max_dist_factor

    neighbors = []
    for i in range(n_patches):
        idxs = [int(j) for j in tree.query_ball_point(centers[i], threshold) if j != i]
        if not idxs and n_patches > 1:
            _, nearest = tree.query(centers[i], k=2)
            idxs = [int(nearest[1])]
        neighbors.append(idxs)
    return neighbors


def load_fault_coordinates(green_dir: str) -> tuple[Tensor, Tensor, list[list[int]]]:
    green_dir_path = _as_path(green_dir)
    mesh = np.load(green_dir_path / "fault_mesh.npz", allow_pickle=True)
    summary = json.loads((green_dir_path / "green_summary.json").read_text())

    centers = mesh["patch_centers"]
    meta = _parse_mesh_meta(mesh["meta"])

    strike_rad = np.radians(meta["strike_deg"])
    strike_vec = np.array([np.sin(strike_rad), np.cos(strike_rad)])

    origin = centers[:, :2].mean(axis=0)
    delta = centers[:, :2] - origin[None, :]
    xi_raw = delta @ strike_vec
    eta_raw = -centers[:, 2]

    xi = (xi_raw - xi_raw.min()) / (xi_raw.max() - xi_raw.min() + 1e-10)
    eta = (eta_raw - eta_raw.min()) / (eta_raw.max() - eta_raw.min() + 1e-10)
    neighbors = build_neighbor_list(
        centers=centers,
        nx=summary.get("nx"),
        ny=summary.get("ny"),
    )
    return (
        torch.tensor(xi, dtype=torch.float32),
        torch.tensor(eta, dtype=torch.float32),
        neighbors,
    )
