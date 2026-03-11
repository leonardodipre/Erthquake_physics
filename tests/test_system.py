from __future__ import annotations

import os
import tempfile
import unittest

import torch

from config import DEFAULT_CONFIG
from dataset import PINNDataloader
from eval import evaluate_gnss_misfit
from fault import load_fault_coordinates
from loss import compute_L_data, compute_L_rsf, compute_L_smooth, compute_L_state
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
        self.assertEqual(len(colloc), 4)
        self.assertTrue(all(loader.t_phys_min <= t <= loader.t_phys_max for t in colloc))

    def test_hybrid_pinn_forward_shapes(self) -> None:
        K_cd, K_ij, summary = load_green_matrices("green_out", device="cpu")
        xi, eta, _ = load_fault_coordinates("green_out")
        model = HybridPINN(K_cd=K_cd, K_ij=K_ij, n_patches=summary["Nc"]).cpu()

        out = model(xi, eta, 1e8)

        self.assertEqual(out["u_surface"].shape, (684,))
        self.assertEqual(out["s"].shape, (1914, 1))
        self.assertEqual(out["V"].shape, (1914, 1))
        self.assertEqual(out["theta"].shape, (1914, 1))
        self.assertEqual(out["tau_elastic"].shape, (1914, 1))
        self.assertEqual(out["tau_rsf"].shape, (1914, 1))

    def test_losses_return_scalars(self) -> None:
        u_pred = torch.ones(6)
        u_obs = torch.zeros(6)
        mask = torch.tensor([1, 1, 0, 1, 0, 1], dtype=torch.float32)
        tau_elastic = torch.ones(4, 1)
        tau_rsf = torch.zeros(4, 1)
        V = torch.full((4, 1), 1e-9)
        theta = torch.full((4, 1), 1e7)
        dtheta_dt = torch.zeros(4, 1)
        d_c = torch.full((4, 1), 1e-2)
        slip = torch.arange(4, dtype=torch.float32).unsqueeze(-1)
        neighbors = [[1], [0, 2], [1, 3], [2]]

        self.assertEqual(compute_L_data(u_pred, u_obs, mask).ndim, 0)
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
            self.assertIn("u_surface_abs_max_m", pred["summary"])
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
            self.assertEqual(len(evaluation["table"]), 2)
        finally:
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
