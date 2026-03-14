"""Diagnostic plots for the PINN earthquake model.

Generates 8 families of plots from a trained checkpoint:
1. Fault slip maps at multiple times
2. Slip rate V maps (log scale)
3. Theta (state variable) maps
4. Friction parameter maps: a, b, D_c, (a-b)
5. Scatter tau_elastic vs tau_rsf + residual map
6. Aging law residual map
7. Time series on selected patches
8. Station-fault sensitivity map / ranking
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch

from predict import load_trained_model
from fault import load_fault_coordinates


def _as_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _load_runtime(checkpoint: str, green_dir: str, device: str | None) -> dict[str, Any]:
    runtime = load_trained_model(
        checkpoint_path=checkpoint, green_dir=green_dir, device=device
    )
    summary = json.loads((_as_path(green_dir) / "green_summary.json").read_text())
    runtime["nx"] = int(summary.get("nx", 1))
    runtime["ny"] = int(summary.get("ny", 1))
    return runtime


def _forward_at(runtime: dict, t_seconds: float) -> dict[str, np.ndarray]:
    model = runtime["model"]
    xi = runtime["xi"]
    eta = runtime["eta"]
    with torch.enable_grad():
        out = model(xi, eta, t_seconds)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def _date_to_seconds(date_str: str, reference_date: str = "2020-01-01") -> float:
    import pandas as pd
    return float((pd.Timestamp(date_str) - pd.Timestamp(reference_date)).total_seconds())


def _time_bound_to_year(value: Any, fallback: float) -> float:
    if value is None:
        return float(fallback)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    import pandas as pd

    try:
        return float(pd.Timestamp(value).year)
    except (TypeError, ValueError):
        return float(fallback)


def _default_time_domain(cfg: dict[str, Any]) -> tuple[float, float]:
    raw = cfg.get("time_domain_physics")
    fallback = (2020.0, 2024.0)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return fallback

    start = _time_bound_to_year(raw[0], fallback[0])
    end = _time_bound_to_year(raw[1], fallback[1])
    if end <= start:
        return fallback
    return start, end


def _prompt_year(label: str, default: float) -> float:
    while True:
        raw = input(f"{label} [{default:g}]: ").strip()
        if not raw:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            print("Inserisci un anno valido, ad esempio 2008.", flush=True)


def _prompt_time_domain(default_start: float, default_end: float) -> tuple[float, float]:
    print("Seleziona gli anni da analizzare. Premi Invio per usare i valori proposti.", flush=True)
    while True:
        start = _prompt_year("Anno iniziale", default_start)
        end = _prompt_year("Anno finale", default_end)
        if end > start:
            return start, end
        print("L'anno finale deve essere maggiore dell'anno iniziale.", flush=True)


def _resolve_time_domain(
    requested: tuple[float | None, float | None] | None,
    cfg: dict[str, Any],
) -> tuple[float, float]:
    start_default, end_default = _default_time_domain(cfg)

    start = start_default
    end = end_default
    needs_prompt = requested is None
    if requested is not None:
        requested_start, requested_end = requested
        if requested_start is not None:
            start = float(requested_start)
        else:
            needs_prompt = True
        if requested_end is not None:
            end = float(requested_end)
        else:
            needs_prompt = True

    if needs_prompt and sys.stdin.isatty():
        start, end = _prompt_time_domain(start, end)

    if end <= start:
        raise ValueError(
            f"Invalid time domain: start={start:g}, end={end:g}. "
            "The end year must be greater than the start year."
        )
    return start, end


def _fault_map(
    values: np.ndarray,
    nx: int,
    ny: int,
    title: str,
    cmap: str = "RdBu_r",
    vmin: float | None = None,
    vmax: float | None = None,
    log_scale: bool = False,
    symmetric: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    grid = values.reshape(ny, nx)
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(12, 4))

    norm = None
    if log_scale:
        grid_pos = np.abs(grid) + 1e-30
        norm = mcolors.LogNorm(vmin=grid_pos[grid_pos > 0].min(), vmax=grid_pos.max())
        im = ax.imshow(grid_pos, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        if symmetric and vmin is None and vmax is None:
            absmax = max(abs(np.nanmin(grid)), abs(np.nanmax(grid)))
            vmin, vmax = -absmax, absmax
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("along-strike (xi index)")
    ax.set_ylabel("down-dip (eta index)")
    return ax


# ── Plot 1: Slip maps ──────────────────────────────────────────────────────────

def plot_slip_maps(
    runtime: dict,
    times_seconds: list[float],
    time_labels: list[str],
    outdir: Path,
    nx: int,
    ny: int,
) -> None:
    n = len(times_seconds)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n))
    if n == 1:
        axes = [axes]
    for ax, t_sec, label in zip(axes, times_seconds, time_labels):
        arrays = _forward_at(runtime, t_sec)
        s = arrays["s"].squeeze()
        _fault_map(s, nx, ny, f"Slip s(t={label}) [m]", cmap="viridis", ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "01_slip_maps.png", dpi=150)
    plt.close(fig)

    # Delta-slip between first and last
    if len(times_seconds) >= 2:
        a0 = _forward_at(runtime, times_seconds[0])["s"].squeeze()
        a1 = _forward_at(runtime, times_seconds[-1])["s"].squeeze()
        fig2, ax2 = plt.subplots(1, 1, figsize=(13, 4))
        _fault_map(
            a1 - a0, nx, ny,
            f"Delta-slip [{time_labels[0]} -> {time_labels[-1]}] [m]",
            cmap="inferno", ax=ax2,
        )
        fig2.tight_layout()
        fig2.savefig(outdir / "01b_delta_slip.png", dpi=150)
        plt.close(fig2)


# ── Plot 2: Slip rate V maps ───────────────────────────────────────────────────

def plot_V_maps(
    runtime: dict,
    times_seconds: list[float],
    time_labels: list[str],
    outdir: Path,
    nx: int,
    ny: int,
) -> None:
    n = len(times_seconds)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n))
    if n == 1:
        axes = [axes]
    for ax, t_sec, label in zip(axes, times_seconds, time_labels):
        arrays = _forward_at(runtime, t_sec)
        V = arrays["V"].squeeze()
        _fault_map(V, nx, ny, f"log10(|V|) at t={label} [m/s]", cmap="hot", log_scale=True, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "02_V_maps.png", dpi=150)
    plt.close(fig)


# ── Plot 3: Theta maps ─────────────────────────────────────────────────────────

def plot_theta_maps(
    runtime: dict,
    times_seconds: list[float],
    time_labels: list[str],
    outdir: Path,
    nx: int,
    ny: int,
) -> None:
    n = len(times_seconds)
    fig, axes = plt.subplots(n, 1, figsize=(13, 3.5 * n))
    if n == 1:
        axes = [axes]
    for ax, t_sec, label in zip(axes, times_seconds, time_labels):
        arrays = _forward_at(runtime, t_sec)
        theta = arrays["theta"].squeeze()
        _fault_map(theta, nx, ny, f"theta at t={label} [s]", cmap="plasma", log_scale=True, ax=ax)
    fig.tight_layout()
    fig.savefig(outdir / "03_theta_maps.png", dpi=150)
    plt.close(fig)


# ── Plot 4: Friction parameter maps ────────────────────────────────────────────

def plot_friction_maps(runtime: dict, outdir: Path, nx: int, ny: int) -> None:
    # Friction params are time-independent (only spatial)
    arrays = _forward_at(runtime, 0.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 8))

    _fault_map(arrays["a"].squeeze(), nx, ny, "a (direct effect)", cmap="YlOrRd", ax=axes[0, 0])
    _fault_map(arrays["b"].squeeze(), nx, ny, "b (evolution effect)", cmap="YlOrRd", ax=axes[0, 1])
    _fault_map(arrays["D_c"].squeeze(), nx, ny, "D_c [m]", cmap="YlGnBu", ax=axes[1, 0])

    ab = (arrays["a"] - arrays["b"]).squeeze()
    _fault_map(ab, nx, ny, "(a - b)", cmap="RdBu_r", symmetric=True, ax=axes[1, 1])

    fig.suptitle("Friction parameters (spatial, time-independent)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "04_friction_maps.png", dpi=150)
    plt.close(fig)


# ── Plot 5: tau_elastic vs tau_rsf scatter + residual map ──────────────────────

def plot_tau_scatter(
    runtime: dict,
    t_seconds: float,
    time_label: str,
    outdir: Path,
    nx: int,
    ny: int,
) -> None:
    arrays = _forward_at(runtime, t_seconds)
    tau_e = arrays["tau_elastic"].squeeze()
    tau_r = arrays["tau_rsf"].squeeze()
    sigma_n = runtime["model"].sigma_n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter
    ax1.scatter(tau_e / 1e6, tau_r / 1e6, s=3, alpha=0.5)
    lims = [
        min(tau_e.min(), tau_r.min()) / 1e6,
        max(tau_e.max(), tau_r.max()) / 1e6,
    ]
    ax1.plot(lims, lims, "k--", lw=1, label="y=x")
    ax1.set_xlabel("tau_elastic [MPa]")
    ax1.set_ylabel("tau_rsf [MPa]")
    ax1.set_title(f"tau_elastic vs tau_rsf at t={time_label}")
    ax1.legend()
    ax1.set_aspect("equal")

    # Normalized residual map
    residual = (tau_e - tau_r) / sigma_n
    _fault_map(
        residual, nx, ny,
        f"RSF residual (tau_e - tau_r) / sigma_n at t={time_label}",
        cmap="RdBu_r", symmetric=True, ax=ax2,
    )

    fig.tight_layout()
    fig.savefig(outdir / "05_tau_scatter.png", dpi=150)
    plt.close(fig)


# ── Plot 6: Aging law residual map ─────────────────────────────────────────────

def plot_aging_residual(
    runtime: dict,
    t_seconds: float,
    time_label: str,
    outdir: Path,
    nx: int,
    ny: int,
) -> None:
    arrays = _forward_at(runtime, t_seconds)
    dtheta_dt = arrays["dtheta_dt"].squeeze()
    V = arrays["V"].squeeze()
    theta = arrays["theta"].squeeze()
    D_c = arrays["D_c"].squeeze()

    aging_target = 1.0 - V * theta / D_c
    residual = dtheta_dt - aging_target

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    _fault_map(
        residual, nx, ny,
        f"Aging law residual (dtheta/dt - target) at t={time_label}",
        cmap="RdBu_r", symmetric=True, ax=ax1,
    )
    _fault_map(
        np.abs(residual), nx, ny,
        f"|Aging law residual| at t={time_label}",
        cmap="hot_r", ax=ax2,
    )

    fig.tight_layout()
    fig.savefig(outdir / "06_aging_residual.png", dpi=150)
    plt.close(fig)


# ── Plot 7: Time series on selected patches ────────────────────────────────────

def _select_representative_patches(nx: int, ny: int, n_patches: int) -> dict[str, int]:
    """Pick patches in key positions: shallow/deep, center/edge."""
    patches = {}
    # Center shallow
    patches["shallow_center"] = 0 * nx + nx // 2
    # Center mid-depth
    patches["mid_center"] = (ny // 2) * nx + nx // 2
    # Center deep
    patches["deep_center"] = (ny - 1) * nx + nx // 2
    # Edge shallow
    patches["shallow_edge"] = 0 * nx + 0
    # Mid-depth, 1/4 along strike
    patches["mid_quarter"] = (ny // 2) * nx + nx // 4
    # Mid-depth, 3/4 along strike
    patches["mid_three_quarter"] = (ny // 2) * nx + (3 * nx) // 4
    # Deep edge
    patches["deep_edge"] = (ny - 1) * nx + nx - 1
    # Clip to valid range
    return {k: min(v, n_patches - 1) for k, v in patches.items()}


def plot_patch_timeseries(
    runtime: dict,
    times_seconds: np.ndarray,
    outdir: Path,
    nx: int,
    ny: int,
    n_patches: int,
) -> None:
    patches = _select_representative_patches(nx, ny, n_patches)

    n_times = len(times_seconds)
    result = {
        "s": np.zeros((n_times, n_patches)),
        "V": np.zeros((n_times, n_patches)),
        "theta": np.zeros((n_times, n_patches)),
        "tau_elastic": np.zeros((n_times, n_patches)),
        "tau_rsf": np.zeros((n_times, n_patches)),
    }

    for i, t_sec in enumerate(times_seconds):
        arrays = _forward_at(runtime, float(t_sec))
        for key in result:
            result[key][i, :] = arrays[key].squeeze()

    days = times_seconds / 86400.0

    fig, axes = plt.subplots(5, 1, figsize=(14, 18), sharex=True)
    titles = ["Slip s [m]", "Slip rate V [m/s]", "State theta [s]",
              "tau_elastic [MPa]", "tau_rsf [MPa]"]
    keys = ["s", "V", "theta", "tau_elastic", "tau_rsf"]
    scales = [1.0, 1.0, 1.0, 1e-6, 1e-6]

    for ax, title, key, scale in zip(axes, titles, keys, scales):
        for name, idx in patches.items():
            vals = result[key][:, idx] * scale
            if key == "V":
                ax.semilogy(days, np.abs(vals), label=f"{name} (#{idx})", lw=0.8)
            else:
                ax.plot(days, vals, label=f"{name} (#{idx})", lw=0.8)
        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=7, ncol=3, loc="best")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time [days from reference]")
    fig.suptitle("Patch time series (representative patches)", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "07_patch_timeseries.png", dpi=150)
    plt.close(fig)


# ── Plot 8: Station sensitivity ─────────────────────────────────────────────────

def plot_station_sensitivity(
    runtime: dict,
    green_dir: str,
    outdir: Path,
) -> None:
    K_cd = runtime["model"].K_cd.detach().cpu().numpy()
    station_ids = np.load(_as_path(green_dir) / "station_ids.npy", allow_pickle=True)
    n_stations = len(station_ids)

    sensitivities = np.zeros(n_stations)
    for i in range(n_stations):
        rows = K_cd[3 * i : 3 * i + 3, :]
        sensitivities[i] = np.linalg.norm(rows)

    order = np.argsort(sensitivities)[::-1]

    fig, ax = plt.subplots(1, 1, figsize=(14, max(6, n_stations * 0.08)))
    y_pos = np.arange(min(n_stations, 50))
    top_order = order[:50]
    ax.barh(y_pos, sensitivities[top_order], color="steelblue", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([str(station_ids[i]) for i in top_order], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("||K_cd(station)|| (sensitivity norm)")
    ax.set_title("Top-50 station sensitivity to fault slip")
    ax.grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    fig.savefig(outdir / "08_station_sensitivity.png", dpi=150)
    plt.close(fig)

    # Save full ranking as JSON
    ranking = [
        {"rank": r + 1, "station": str(station_ids[i]), "sensitivity": float(sensitivities[i])}
        for r, i in enumerate(order)
    ]
    with open(outdir / "08_station_sensitivity_ranking.json", "w") as f:
        json.dump(ranking, f, indent=2)


# ── Plot 9: Station geometric weights map ────────────────────────────────────

def plot_station_geo_weights(
    green_dir: str,
    stations_json: str,
    outdir: Path,
    k: int = 5,
    alpha: float = 0.5,
    clip_min: float = 0.5,
    clip_max: float = 2.0,
) -> None:
    """Plot station positions colored by geometric density weight."""
    from dataset import compute_station_geo_weights

    station_ids_path = str(_as_path(green_dir) / "station_ids.npy")
    _, diagnostics = compute_station_geo_weights(
        station_ids_path=station_ids_path,
        stations_json_path=stations_json,
        k=k, alpha=alpha, clip_min=clip_min, clip_max=clip_max,
    )

    lats = np.array([d["lat"] for d in diagnostics])
    lons = np.array([d["lon"] for d in diagnostics])
    weights = np.array([d["weight"] for d in diagnostics])
    names = [d["station"] for d in diagnostics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # Map view
    sc = ax1.scatter(lons, lats, c=weights, cmap="RdYlGn", s=40,
                     edgecolors="k", linewidths=0.3, vmin=clip_min, vmax=clip_max)
    plt.colorbar(sc, ax=ax1, label="Station weight", shrink=0.8)
    ax1.set_xlabel("Longitude [°E]")
    ax1.set_ylabel("Latitude [°N]")
    ax1.set_title("Geometric station weights (green=upweighted, red=downweighted)")
    ax1.set_aspect(1.0 / np.cos(np.radians(np.mean(lats))))
    ax1.grid(True, alpha=0.3)

    # Label stations with extreme weights
    for i, (lon, lat, w, name) in enumerate(zip(lons, lats, weights, names)):
        if w <= clip_min + 0.05 or w >= clip_max - 0.05:
            ax1.annotate(name[:8], (lon, lat), fontsize=5, alpha=0.7,
                         xytext=(2, 2), textcoords="offset points")

    # Histogram
    ax2.hist(weights, bins=30, color="steelblue", edgecolor="white", linewidth=0.5)
    ax2.axvline(1.0, color="k", linestyle="--", lw=1, label="mean=1")
    ax2.set_xlabel("Station weight")
    ax2.set_ylabel("Count")
    ax2.set_title("Weight distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(outdir / "09_station_geo_weights.png", dpi=150)
    plt.close(fig)
    print(f"Station geo-weight plot saved to {outdir / '09_station_geo_weights.png'}", flush=True)


# ── Plot 10: Training history ─────────────────────────────────────────────────

def plot_training_history(history_csv: str, outdir: Path) -> None:
    """Plot training loss curves, (a-b) stats, V stats, and LR from history CSV."""
    import pandas as pd

    df = pd.read_csv(history_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    steps = df["step"].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: Loss terms (log scale)
    ax = axes[0, 0]
    loss_cols = [c for c in df.columns if c.startswith("L_") and c != "L_total"]
    ax.semilogy(steps, df["L_total"].values, "k-", lw=1.5, label="L_total")
    for col in loss_cols:
        if col in df.columns:
            vals = df[col].values
            if np.any(vals > 0):
                ax.semilogy(steps, vals, lw=0.8, label=col)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (log)")
    ax.set_title("Loss terms")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # Panel 2: (a-b) statistics
    ax = axes[0, 1]
    if "a_minus_b_mean" in df.columns:
        ax.plot(steps, df["a_minus_b_mean"].values, label="mean", lw=1)
        ax.fill_between(
            steps,
            df["a_minus_b_mean"].values - df["a_minus_b_std"].values,
            df["a_minus_b_mean"].values + df["a_minus_b_std"].values,
            alpha=0.2, label="mean +/- std",
        )
        ax.plot(steps, df["a_minus_b_min"].values, "--", lw=0.6, label="min")
        ax.plot(steps, df["a_minus_b_max"].values, "--", lw=0.6, label="max")
    ax.set_xlabel("Step")
    ax.set_ylabel("(a - b)")
    ax.set_title("Friction parameter (a-b)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Panel 3: V statistics
    ax = axes[1, 0]
    if "V_log10_mean" in df.columns:
        ax.plot(steps, df["V_log10_mean"].values, label="log10(|V|) mean", lw=1)
        ax.fill_between(
            steps,
            df["V_log10_mean"].values - df["V_log10_std"].values,
            df["V_log10_mean"].values + df["V_log10_std"].values,
            alpha=0.2, label="mean +/- std",
        )
    if "slip_max" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(steps, df["slip_max"].values, "r-", lw=0.8, label="slip_max [m]")
        ax2.set_ylabel("slip_max [m]", color="r")
        ax2.legend(fontsize=7, loc="upper right")
    ax.set_xlabel("Step")
    ax.set_ylabel("log10(|V|)")
    ax.set_title("Slip rate V + max slip")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Panel 4: LR + warmup weights
    ax = axes[1, 1]
    if "lr" in df.columns:
        ax.plot(steps, df["lr"].values, "b-", lw=1, label="learning rate")
        ax.set_ylabel("LR", color="b")
    if "w_rsf" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(steps, df["w_rsf"].values, "g-", lw=0.8, label="w_rsf")
        ax2.plot(steps, df["w_state"].values, "m-", lw=0.8, label="w_state")
        ax2.set_ylabel("Warmup weight")
        ax2.legend(fontsize=7, loc="center right")
    ax.set_xlabel("Step")
    ax.set_title("LR schedule + physics warmup")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Training history: {Path(history_csv).name}", fontsize=13)
    fig.tight_layout()
    fig.savefig(outdir / "10_training_history.png", dpi=150)
    plt.close(fig)
    print(f"Training history plot saved to {outdir / '10_training_history.png'}", flush=True)


# ── Main entry ──────────────────────────────────────────────────────────────────

def run_diagnostics(
    checkpoint: str,
    green_dir: str = "green_out",
    outdir: str = "diagnostics",
    device: str | None = None,
    time_domain: tuple[float | None, float | None] | None = None,
    n_time_samples: int = 30,
    reference_date: str = "2020-01-01",
) -> None:
    outdir_path = _as_path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {checkpoint}...", flush=True)
    runtime = _load_runtime(checkpoint, green_dir, device)
    nx = runtime["nx"]
    ny = runtime["ny"]
    n_patches = nx * ny
    cfg = runtime["config"]
    ref_date = str(cfg.get("reference_date", reference_date))
    start_year, end_year = _resolve_time_domain(time_domain, cfg)
    print(f"Analisi diagnostica su intervallo {start_year:g} -> {end_year:g}", flush=True)

    # Build time grid
    t_start = _date_to_seconds(f"{int(start_year)}-01-01", ref_date)
    t_end = _date_to_seconds(f"{int(end_year)}-01-01", ref_date)

    # Snapshot times: start, 1/3, 2/3, end
    snapshot_seconds = [
        t_start,
        t_start + (t_end - t_start) / 3,
        t_start + 2 * (t_end - t_start) / 3,
        t_end,
    ]
    import pandas as pd
    snapshot_labels = [
        str((pd.Timestamp(ref_date) + pd.Timedelta(seconds=t)).date())
        for t in snapshot_seconds
    ]

    # Dense time grid for patch time series
    dense_times = np.linspace(t_start, t_end, n_time_samples)

    # Mid-point for single-time plots
    t_mid = t_start + (t_end - t_start) / 2
    mid_label = str((pd.Timestamp(ref_date) + pd.Timedelta(seconds=t_mid)).date())

    print("Generating plots...", flush=True)

    print("  [1/9] Slip maps...", flush=True)
    plot_slip_maps(runtime, snapshot_seconds, snapshot_labels, outdir_path, nx, ny)

    print("  [2/9] Slip rate V maps...", flush=True)
    plot_V_maps(runtime, snapshot_seconds, snapshot_labels, outdir_path, nx, ny)

    print("  [3/9] Theta maps...", flush=True)
    plot_theta_maps(runtime, snapshot_seconds, snapshot_labels, outdir_path, nx, ny)

    print("  [4/9] Friction parameter maps...", flush=True)
    plot_friction_maps(runtime, outdir_path, nx, ny)

    print("  [5/9] tau_elastic vs tau_rsf...", flush=True)
    plot_tau_scatter(runtime, t_mid, mid_label, outdir_path, nx, ny)

    print("  [6/9] Aging law residual...", flush=True)
    plot_aging_residual(runtime, t_mid, mid_label, outdir_path, nx, ny)

    print("  [7/9] Patch time series...", flush=True)
    plot_patch_timeseries(runtime, dense_times, outdir_path, nx, ny, n_patches)

    print("  [8/9] Station sensitivity...", flush=True)
    plot_station_sensitivity(runtime, green_dir, outdir_path)

    print("  [9/9] Station geo weights...", flush=True)
    stations_json = str(cfg.get("stations_json", "dataset_scremato/stations_ITA_laquila_150km.json"))
    plot_station_geo_weights(
        green_dir=green_dir,
        stations_json=stations_json,
        outdir=outdir_path,
        k=int(cfg.get("geo_weight_k", 5)),
        alpha=float(cfg.get("geo_weight_alpha", 0.5)),
        clip_min=float(cfg.get("geo_weight_clip_min", 0.5)),
        clip_max=float(cfg.get("geo_weight_clip_max", 2.0)),
    )

    print(f"All diagnostics saved to {outdir_path}/", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate diagnostic plots for trained PINN")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--green-dir", default="green_out")
    parser.add_argument("--outdir", default="diagnostics")
    parser.add_argument("--device", default=None)
    parser.add_argument("--time-start", type=float, default=None)
    parser.add_argument("--time-end", type=float, default=None)
    parser.add_argument("--n-time-samples", type=int, default=30)
    parser.add_argument("--history", default=None, help="Path to training history CSV")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.history:
        outdir_path = _as_path(args.outdir)
        outdir_path.mkdir(parents=True, exist_ok=True)
        plot_training_history(args.history, outdir_path)
        return

    run_diagnostics(
        checkpoint=args.checkpoint,
        green_dir=args.green_dir,
        outdir=args.outdir,
        device=args.device,
        time_domain=(
            args.time_start,
            args.time_end,
        ),
        n_time_samples=args.n_time_samples,
    )


if __name__ == "__main__":
    main()
