import unittest

import torch

from us_imaging.models.rf_autoencoder import MODEL_REGISTRY, build_rf_autoencoder
from us_imaging.models.rf_transformer import RFHybridTransformerAE
from us_imaging.models.train_ae import combined_loss


class RFHybridTransformerTests(unittest.TestCase):
    def test_hybrid_registered(self):
        self.assertIs(MODEL_REGISTRY["hybrid"], RFHybridTransformerAE)

    def test_hybrid_single_channel_shape(self):
        model = build_rf_autoencoder("hybrid", input_len=256, latent_dim=64, in_channels=1)
        model.eval()
        x = torch.randn(2, 1, 256)

        with torch.no_grad():
            recon, latent = model(x)

        self.assertEqual(tuple(recon.shape), (2, 1, 256))
        self.assertEqual(tuple(latent.shape), (2, 64))
        self.assertEqual(model.token_len, 64)

    def test_hybrid_three_channel_shape_and_loss(self):
        model = build_rf_autoencoder("hybrid", input_len=256, latent_dim=64, in_channels=3)
        model.eval()
        x = torch.randn(2, 3, 256)

        with torch.no_grad():
            recon, latent = model(x)
            loss, comps = combined_loss(x, recon, lambda_freq=0.1, lambda_coherence=0.05)

        self.assertEqual(tuple(recon.shape), (2, 3, 256))
        self.assertEqual(tuple(latent.shape), (2, 64))
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("coherence", comps)
        self.assertGreaterEqual(comps["coherence"], 0.0)


if __name__ == "__main__":
    unittest.main()
