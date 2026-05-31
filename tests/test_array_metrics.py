import unittest

import numpy as np

from us_imaging.simulation.acquire import TransducerArray, array_engineering_metrics


class ArrayEngineeringMetricsTests(unittest.TestCase):
    def test_default_array_metrics(self):
        metrics = array_engineering_metrics(TransducerArray())

        self.assertAlmostEqual(metrics["wavelength_mm"], 0.308, places=3)
        self.assertAlmostEqual(metrics["pitch_over_lambda"], 0.974, places=3)
        self.assertAlmostEqual(metrics["depth_sample_spacing_um"], 19.25, places=2)
        self.assertAlmostEqual(metrics["nyquist_mhz"], 20.0, places=3)
        self.assertAlmostEqual(metrics["aperture_mm"], 18.9, places=3)
        self.assertAlmostEqual(metrics["axial_resolution_mm"], 0.22, places=3)

    def test_cross_beam_correlation_metrics(self):
        base = np.array([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
        patches = np.stack(
            [
                np.stack([base, base, base], axis=0),
                np.stack([base, base, base], axis=0),
            ],
            axis=0,
        )

        metrics = array_engineering_metrics(TransducerArray(), patches=patches)

        self.assertAlmostEqual(metrics["cross_beam_corr_mean"], 1.0, places=6)
        self.assertAlmostEqual(metrics["cross_beam_corr_std"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
