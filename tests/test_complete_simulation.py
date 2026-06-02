import os
import tempfile
import unittest

import numpy as np

from us_imaging.simulation.complete_simulation import (
    CompleteSimulationConfig,
    run_complete_simulation,
    save_complete_simulation_report,
)
from us_imaging.simulation.micro_array import get_micro_array_config


class CompleteSimulationTests(unittest.TestCase):
    def test_complete_simulation_outputs_shapes_and_metrics(self):
        config = CompleteSimulationConfig(
            micro_array_config=get_micro_array_config(
                "default",
                n_elements=16,
                receive_noise_snr_db=40.0,
            ),
            n_point=2,
            n_dense=20,
            n_beams=3,
            x_pixels=12,
            z_pixels=16,
            base_seed=101,
        )

        result = run_complete_simulation(config)

        self.assertEqual(result["rf_data"].ndim, 3)
        self.assertEqual(result["patches"].shape[1:], (3, 256))
        self.assertEqual(result["beamformed"].shape, (16, 12))
        self.assertEqual(result["bmode"].shape, (16, 12))
        self.assertTrue(np.all(np.isfinite(result["bmode"])))
        self.assertGreaterEqual(result["metrics"]["n_patches"], 1)
        self.assertEqual(
            result["metrics"]["simulation_layers"],
            "electroacoustic,propagation,receive_chain,beamforming",
        )

    def test_complete_simulation_report_writes_expected_files(self):
        config = CompleteSimulationConfig(
            micro_array_config=get_micro_array_config(
                "default",
                n_elements=16,
                receive_noise_snr_db=None,
            ),
            n_point=2,
            n_dense=20,
            n_beams=3,
            x_pixels=8,
            z_pixels=12,
            base_seed=202,
        )
        result = run_complete_simulation(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_complete_simulation_report(result, tmpdir)

            self.assertTrue(os.path.exists(os.path.join(tmpdir, "complete_simulation_metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "rf_data.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "patches.npy")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "bmode.npy")))


if __name__ == "__main__":
    unittest.main()
