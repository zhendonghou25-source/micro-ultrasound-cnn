import unittest

import torch

from us_imaging.models.rf_autoencoder import (
    MODEL_REGISTRY,
    RFAutoencoderV6,
    build_rf_autoencoder,
)


class RFAutoencoderShapeTests(unittest.TestCase):
    def test_single_channel_forward_shapes_v1_to_v6(self):
        for version in sorted(MODEL_REGISTRY):
            with self.subTest(version=version):
                model = build_rf_autoencoder(
                    version,
                    input_len=256,
                    latent_dim=64,
                    in_channels=1,
                )
                model.eval()
                x = torch.randn(2, 1, 256)
                with torch.no_grad():
                    recon, latent = model(x)

                self.assertEqual(tuple(recon.shape), (2, 1, 256))
                self.assertEqual(tuple(latent.shape), (2, 64))
                self.assertGreater(model.count_parameters(), 0)

    def test_multi_channel_forward_shapes_v2_to_v6(self):
        for version in ["v2", "v3", "v4", "v5", "v6"]:
            with self.subTest(version=version):
                model = build_rf_autoencoder(
                    version,
                    input_len=256,
                    latent_dim=64,
                    in_channels=3,
                )
                model.eval()
                x = torch.randn(2, 3, 256)
                with torch.no_grad():
                    recon, latent = model(x)

                self.assertEqual(tuple(recon.shape), (2, 3, 256))
                self.assertEqual(tuple(latent.shape), (2, 64))

    def test_v1_rejects_multi_channel_input_in_factory(self):
        with self.assertRaises(ValueError):
            build_rf_autoencoder("v1", input_len=256, latent_dim=64, in_channels=3)

    def test_v6_is_registered(self):
        self.assertIs(MODEL_REGISTRY["v6"], RFAutoencoderV6)


if __name__ == "__main__":
    unittest.main()
