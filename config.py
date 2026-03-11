from __future__ import annotations

from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "slip_hidden": 64,
    "slip_blocks": 4,
    "friction_hidden": 32,
    "friction_blocks": 3,
    "dropout": 0.05,
    "mu_0": 0.6,
    "V_0": 1e-9,
    "sigma_n": 100e6,
    "lambda_data": 1.0,
    "lambda_rsf": 0.05,
    "lambda_state": 0.05,
    "lambda_smooth": 1e-3,
    "total_steps": 40_000,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "weight_decay": 1e-4,
    "grad_clip": 1.0,
    "data_batch_size": 1,
    "n_colloc_per_step": 8,
    "time_ranges_data": [(2000, 2008)],
    "time_domain_physics": (2000, 2008),
    "reference_date": "2000-01-01",
    "log_every": 100,
    "tau_dot_init": 0.1,
    "time_input_scale": 1e8,
}
