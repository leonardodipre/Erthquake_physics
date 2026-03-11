from __future__ import annotations

import os
import tempfile
import unittest

import torch

from config import DEFAULT_CONFIG
from dataset import PINNDataloader, list_markers_in_subdir
from eval import evaluate_gnss_misfit
from eval_holdout import evaluate_holdout_stations
from fault import load_fault_coordinates
from loss import compute_L_data, compute_L_rsf, compute_L_smooth, compute_L_state, compute_L_velocity
from model import HybridPINN, load_green_matrices
from predict import predict_snapshot


class TestPINNBranchB(unittest.TestCase):
    def test_green_shapes_match_summary(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        self.assertEqual(K_cd.shape, (684, summary["Nc"]))
        self.assertEqual(K_ij.shape, (summary["Nc"], summary["Nc"]))

    def test_fault_coordinates_and_neighbors(self) -> None:
        xi, eta, neighbors = load_fault_coordinates("green_out")
        self.assertEqual(xi.shape, (1914,))
        self.assertEqual(eta.shape, (1914,))
        self.assertEqual(len(neighbors), 1914)
        self.assertTrue(all(len(nbrs) > 0 for nbrs in neighbors))

    def test_dataset_dual_sampler_shapes(self) -> None:
        loader = PINNDataloader(
            data_dir="dataset_scremato",
            station_ids_path="green_out/station_ids.npy",
            time_ranges_data=DEFAULT_CONFIG["time_ranges_data"],
            time_domain_physics=DEFAULT_CONFIG["time_domain_physics"],
            reference_date=DEFAULT_CONFIG["reference_date"],
            seed=7,
        )
        sample = loader.sample_data_batch(batch_size=1)[0]
        colloc = loader.sample_collocation_batch(batch_size=4)

        self.assertEqual(sample["u_observed"].shape, (684,))
        self.assertEqual(sample["mask_data"].shape, (684,))
        self.assertEqual(sample["v_observed"].shape, (684,))
        self.assertEqual(sample["mask_velocity"].shape, (684,))
        self.assertGreaterEqual(int(sample["mask_velocity"].sum().item()), 0)
        self.assertEqual(len(colloc), 4)
        self.assertTrue(all(loader.t_phys_min <= t <= loader.t_phys_max for t in colloc))

    def test_dataset_can_exclude_holdout_markers(self) -> None:
        holdout_markers = list_markers_in_subdir("dataset_scremato", "acc_test")
        loader = PINNDataloader(
            data_dir="dataset_scremato",
            station_ids_path="green_out/station_ids.npy",
            time_ranges_data=DEFAULT_CONFIG["time_ranges_data"],
            time_domain_physics=DEFAULT_CONFIG["time_domain_physics"],
            reference_date=DEFAULT_CONFIG["reference_date"],
            excluded_markers=holdout_markers,
            seed=7,
        )
        sample = loader.sample_data_batch(batch_size=1)[0]
        holdout_mask = loader.holdout_component_mask.bool()

        self.assertEqual(int(loader.holdout_component_mask.sum().item()), 3 * len(holdout_markers))
        self.assertEqual(float(sample["mask_data"][holdout_mask].sum().item()), 0.0)
        self.assertEqual(float(sample["mask_velocity"][holdout_mask].sum().item()), 0.0)

    def test_hybrid_pinn_forward_shapes(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        xi, eta, _ = load_fault_coordinates("green_out")
        model = HybridPINN(K_cd=K_cd, K_ij=K_ij, n_patches=summary["Nc"]).cpu()

        out = model(xi, eta, 1e8)

        self.assertEqual(out["u_surface"].shape, (684,))
        self.assertEqual(out["v_surface"].shape, (684,))
        self.assertEqual(out["s"].shape, (1914, 1))
        self.assertEqual(out["V"].shape, (1914, 1))
        self.assertEqual(out["theta"].shape, (1914, 1))
        self.assertEqual(out["tau_elastic"].shape, (1914, 1))
        self.assertEqual(out["tau_rsf"].shape, (1914, 1))

    def test_losses_return_scalars(self) -> None:
        u_pred = torch.ones(6)
        u_obs = torch.zeros(6)
        mask = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.float32)
        v_pred = torch.full((6,), 2e-10)
        v_obs = torch.zeros(6)
        tau_elastic = torch.ones(4, 1)
        tau_rsf = torch.zeros(4, 1)
        V = torch.full((4, 1), 1e-9)
        theta = torch.full((4, 1), 1e7)
        dtheta_dt = torch.zeros(4, 1)
        d_c = torch.full((4, 1), 1e-2)
        slip = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
        neighbors = [[1], [0, 2], [1, 3], [2]]

        self.assertEqual(compute_L_data(u_pred, u_obs, mask).ndim, 0)
        self.assertEqual(compute_L_velocity(v_pred, v_obs, mask).ndim, 0)
        self.assertEqual(compute_L_rsf(tau_elastic, tau_rsf, sigma_n=1e8).ndim, 0)
        self.assertEqual(compute_L_state(dtheta_dt, V, theta, d_c).ndim, 0)
        self.assertEqual(compute_L_smooth(slip, neighbors).ndim, 0)

    def test_predict_snapshot_from_checkpoint(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        model = HybridPINN(K_cd=K_cd, K_ij=K_ij, n_patches=summary["Nc"]).cpu()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": dict(DEFAULT_CONFIG),
                    "green_dir": "green_out",
                },
                checkpoint_path,
            )
            pred = predict_snapshot(
                checkpoint_path=checkpoint_path,
                green_dir="green_out",
                device="cpu",
                time_seconds=1e8,
            )
            self.assertEqual(pred["arrays"]["u_surface"].shape, (684,))
            self.assertEqual(pred["arrays"]["v_surface"].shape, (684,))
            self.assertIn("u_surface_abs_max_m", pred["summary"])
            self.assertIn("v_surface_abs_max_mm_per_day", pred["summary"])
        finally:
            os.unlink(checkpoint_path)

    def test_eval_gnss_misfit_from_checkpoint(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        model = HybridPINN(K_cd=K_cd, K_ij=K_ij, n_patches=summary["Nc"]).cpu()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": dict(DEFAULT_CONFIG),
                    "green_dir": "green_out",
                    "data_dir": "dataset_scremato",
                },
                checkpoint_path,
            )
            evaluation = evaluate_gnss_misfit(
                checkpoint_path=checkpoint_path,
                green_dir="green_out",
                data_dir="dataset_scremato",
                device="cpu",
                max_samples=2,
            )
            self.assertEqual(evaluation["summary"]["n_time_samples"], 2)
            self.assertIn("global_rmse_mm", evaluation["summary"])
            self.assertIn("global_velocity_rmse_mm_per_day", evaluation["summary"])
            self.assertEqual(len(evaluation["table"]), 2)
        finally:
            os.unlink(checkpoint_path)

    def test_eval_holdout_from_checkpoint(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        model = HybridPINN(K_cd=K_cd, K_ij=K_ij, n_patches=summary["Nc"]).cpu()

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
            checkpoint_path = tmp.name

        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": dict(DEFAULT_CONFIG),
                    "green_dir": "green_out",
                    "data_dir": "dataset_scremato",
                },
                checkpoint_path,
            )
            evaluation = evaluate_holdout_stations(
                checkpoint_path=checkpoint_path,
                green_dir="green_out",
                data_dir="dataset_scremato",
                device="cpu",
                markers_subdir="acc_test",
                days=2,
                max_samples=2,
                output_dir=None,
                make_plots=False,
            )
            self.assertEqual(evaluation["summary"]["n_time_samples"], 2)
            self.assertIn("global_holdout_rmse_mm", evaluation["summary"])
            self.assertEqual(len(evaluation["table"]), 2)
        finally:
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
