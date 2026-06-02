import unittest

from us_imaging.evaluation.cnn_baseline_audit import (
    architecture_audit_rows,
    compute_receptive_field,
    summarize_kernel_frequency,
    LayerSpec,
)
from us_imaging.models.rf_autoencoder import build_rf_autoencoder


class CNNBaselineAuditTests(unittest.TestCase):
    def test_receptive_field_formula(self):
        rf, stride = compute_receptive_field([
            LayerSpec(kernel_size=3, stride=2),
            LayerSpec(kernel_size=5, stride=2),
        ])

        self.assertEqual(rf, 11)
        self.assertEqual(stride, 4)

    def test_architecture_rows_cover_v1_to_v6(self):
        rows = architecture_audit_rows(in_channels=3)
        by_version = {row["model_version"]: row for row in rows}

        self.assertEqual(set(by_version), {"v1", "v2", "v3", "v4", "v5", "v6"})
        self.assertFalse(by_version["v1"]["supports_multichannel"])
        self.assertTrue(by_version["v6"]["supports_multichannel"])
        self.assertEqual(by_version["v1"]["model_in_channels"], 1)
        self.assertEqual(by_version["v6"]["model_in_channels"], 3)
        self.assertGreater(by_version["v6"]["parameters"], by_version["v2"]["parameters"])
        self.assertGreater(by_version["v6"]["receptive_field_max_depth_mm"], 0.0)

    def test_kernel_summary_for_v6_frontend(self):
        model = build_rf_autoencoder("v6", in_channels=3)
        rows = summarize_kernel_frequency(model)
        layer_names = {row["layer_name"] for row in rows}

        self.assertEqual(layer_names, {"branch_small", "branch_med", "branch_large"})
        self.assertTrue(all(row["peak_freq_mhz"] >= 0.0 for row in rows))


if __name__ == "__main__":
    unittest.main()
