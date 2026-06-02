import unittest

import numpy as np

from us_imaging.simulation.micro_array import (
    apply_receive_chain,
    generate_parameterized_multibeam_data,
    get_micro_array_config,
    micro_array_metrics,
    parameterized_plane_wave_rf,
    sample_element_response,
)
from us_imaging.simulation.phantom import Phantom


class MicroArrayModelTests(unittest.TestCase):
    def test_templates_and_metrics_are_precise(self):
        config = get_micro_array_config("pmut_like")
        metrics = micro_array_metrics(config)

        self.assertEqual(config.array_type, "pmut_like")
        self.assertAlmostEqual(metrics["center_freq_mhz"], 16.0, places=6)
        self.assertAlmostEqual(metrics["pitch_mm"], 0.1, places=6)
        self.assertAlmostEqual(metrics["q_factor"], 2.2, places=6)
        self.assertIn("wavelength_mm", metrics)
        self.assertIn("pitch_over_lambda", metrics)

    def test_element_response_sampling(self):
        config = get_micro_array_config(
            "default",
            n_elements=8,
            element_failure_rate=0.25,
            receive_noise_snr_db=None,
        )
        response = sample_element_response(config, seed=123)

        self.assertEqual(response["amp_factors"].shape, (8,))
        self.assertEqual(response["phase_offsets_rad"].shape, (8,))
        self.assertEqual(response["center_freq_offsets_hz"].shape, (8,))
        self.assertEqual(response["active_mask"].shape, (8,))
        self.assertLess(np.sum(response["active_mask"]), 8)

    def test_receive_chain_preserves_shape_and_finite_values(self):
        config = get_micro_array_config(
            "default",
            n_elements=8,
            receive_noise_snr_db=40.0,
        )
        rf = np.ones((1, 8, 32), dtype=np.float32)
        out, response = apply_receive_chain(rf, config, seed=7)

        self.assertEqual(out.shape, rf.shape)
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertIn("active_mask", response)

    def test_parameterized_plane_wave_rf_shape(self):
        config = get_micro_array_config(
            "default",
            n_elements=8,
            receive_noise_snr_db=None,
        )
        phantom = Phantom.point_target(x_m=0.0, z_m=0.01)
        rf, metrics = parameterized_plane_wave_rf(phantom, config, seed=5)

        self.assertEqual(rf.ndim, 3)
        self.assertEqual(rf.shape[0], 1)
        self.assertEqual(rf.shape[1], 8)
        self.assertTrue(np.all(np.isfinite(rf)))
        self.assertEqual(metrics["active_elements"], 8)

    def test_parameterized_multibeam_data_shape_and_metrics(self):
        config = get_micro_array_config(
            "default",
            n_elements=16,
            receive_noise_snr_db=35.0,
        )
        patches, labels, metrics = generate_parameterized_multibeam_data(
            n_phantoms=1,
            config=config,
            n_point_per_phantom=2,
            n_dense=20,
            n_beams=3,
            patch_len=256,
            base_seed=11,
        )

        self.assertEqual(patches.ndim, 3)
        self.assertEqual(patches.shape[1:], (3, 256))
        self.assertEqual(len(labels), len(patches))
        self.assertTrue(np.all(np.isfinite(patches)))
        self.assertIn("cross_beam_corr_mean", metrics)
        self.assertTrue(np.isfinite(metrics["cross_beam_corr_mean"]))


if __name__ == "__main__":
    unittest.main()
