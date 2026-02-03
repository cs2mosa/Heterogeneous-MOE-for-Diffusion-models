import unittest
import torch
import pickle
from Utils.VAE_CLIP import StabilityVAE, CLIP_EMBED

class TestStabilityVAEEncoder(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests in this class."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  -> Testing VAE on device: {cls.device}")

    def setUp(self):
        """Runs before EACH test method."""
        self.vae_encoder = StabilityVAE()
        # Create a dummy image: Batch=2, Channels=3, 64x64 resolution
        self.dummy_image = torch.randint(0, 256, (2, 3, 64, 64), device=self.device, dtype=torch.uint8)

    def test_lazy_loading(self):
        """Ensure the VAE model is not loaded until used."""
        self.assertIsNone(self.vae_encoder._vae, "VAE should be None on init")

        # Trigger load
        _ = self.vae_encoder.encode(self.dummy_image)

        self.assertIsNotNone(self.vae_encoder._vae, "VAE should be loaded after encode call")
        self.assertFalse(self.vae_encoder._vae.training, "VAE should be in eval mode")

    def test_shape_and_dtype(self):
        """Ensure input/output shapes and types are correct."""
        latents = self.vae_encoder.encode(self.dummy_image)

        # Check Latents
        expected_shape = (2, 4, 8, 8)  # 64 / 8 = 8
        self.assertEqual(latents.shape, expected_shape)
        self.assertTrue(latents.dtype in [torch.float32, torch.float16, torch.bfloat16])

        # Check Decoding
        decoded = self.vae_encoder.decode(latents)
        self.assertEqual(decoded.shape, self.dummy_image.shape)
        self.assertEqual(decoded.dtype, torch.uint8)

    def test_lossy_reconstruction(self):
        """VAE is lossy; output should not equal input exactly."""
        latents = self.vae_encoder.encode(self.dummy_image)
        decoded = self.vae_encoder.decode(latents)

        # Should not be exactly equal
        self.assertFalse(torch.equal(self.dummy_image, decoded))

        # But should be reasonably close (checking MSE on float)
        mse = torch.nn.functional.mse_loss(self.dummy_image.float(), decoded.float())
        # Note: Random noise image reconstruction MSE is high, but we just want to ensure code runs
        self.assertGreater(mse.item(), 0.0)

    def test_serialization(self):
        """Ensure pickling excludes the heavy VAE model."""
        # Force load the VAE first
        self.vae_encoder.init(self.device)
        self.assertIsNotNone(self.vae_encoder._vae)

        # Pickle
        pickled_data = pickle.dumps(self.vae_encoder)

        # Unpickle
        restored_encoder = pickle.loads(pickled_data)

        # Check that heavy model is gone
        self.assertIsNone(restored_encoder._vae, "Restored encoder should have None for _vae")

        # Check that it still works (auto-reloads)
        latents = restored_encoder.encode(self.dummy_image)
        self.assertEqual(latents.shape, (2, 4, 8, 8))


class TestCLIPEmbed(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  -> Testing CLIP on device: {cls.device}")

    def setUp(self):
        self.clip_embed = CLIP_EMBED(device=self.device)
        self.prompts = ["a photo of a cat", "short", "a very long description " * 10]

    def test_lazy_loading(self):
        """Ensure CLIP model is not loaded until used."""
        self.assertIsNone(self.clip_embed.text_encoder)
        self.assertIsNone(self.clip_embed.tokenizer)

        _ = self.clip_embed.encode_text(["test"])

        self.assertIsNotNone(self.clip_embed.text_encoder)
        self.assertIsNotNone(self.clip_embed.tokenizer)

    def test_shape_and_padding(self):
        """Ensure outputs are [B, 77, 768] regardless of input length."""
        embeddings = self.clip_embed.encode_text(self.prompts)

        expected_shape = (3, 77, 768)  # 3 prompts
        self.assertEqual(embeddings.shape, expected_shape)

    def test_determinism(self):
        """Ensure the same text produces the exact same tensor (eval mode)."""
        text = ["deterministic test"]
        embed1 = self.clip_embed.encode_text(text)
        embed2 = self.clip_embed.encode_text(text)

        self.assertTrue(torch.equal(embed1, embed2), "Embeddings should be deterministic")

    def test_serialization(self):
        """Ensure pickling excludes the heavy CLIP model."""
        self.clip_embed.init()
        self.assertIsNotNone(self.clip_embed.text_encoder)

        pickled_data = pickle.dumps(self.clip_embed)
        restored_embedder = pickle.loads(pickled_data)

        self.assertIsNone(restored_embedder.text_encoder)
        self.assertIsNone(restored_embedder.tokenizer)

        # Check functionality
        embeds = restored_embedder.encode_text(["test"])
        self.assertEqual(embeds.shape, (1, 77, 768))


if __name__ == '__main__':
    unittest.main()