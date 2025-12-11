import torch
import unittest
from models import model_internals as m
import torch.nn as nn
class TestEmbeddings(unittest.TestCase):

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 8

        # --- Config for Pos_encoding ---
        self.pos_emb_dim = 128
        self.pos_freq_dim = 64
        self.pos_model = m.Pos_encoding(
            emb_dim=self.pos_emb_dim,
            freq_emb_dim=self.pos_freq_dim
        ).to(self.device)

        # --- Config for MP_Fourier ---
        self.four_channels = 256
        self.fourier_model = m.MP_Fourier(
            num_channels=self.four_channels,
            bandwidth=1.5
        ).to(self.device)

    # ==========================================
    # TESTS FOR POS_ENCODING
    # ==========================================

    def test_pos_output_shape(self):
        """[PosEncoding] Output shape match"""
        t = torch.randint(0, 1000, (self.batch_size,)).to(self.device)
        out = self.pos_model(t)
        self.assertEqual(out.shape, (self.batch_size, self.pos_emb_dim))

    def test_pos_input_robustness(self):
        """[PosEncoding] Handle (B,) and (B,1) inputs identically"""
        t_flat = torch.randint(0, 1000, (self.batch_size,)).to(self.device)
        t_2d = t_flat.unsqueeze(1)

        out_flat = self.pos_model(t_flat)
        out_2d = self.pos_model(t_2d)

        self.assertTrue(torch.allclose(out_flat, out_2d),
                        "PosEncoding output differed for (B,) vs (B,1) input")

    def test_pos_gradient_flow(self):
        """[PosEncoding] MLP weights should be trainable"""
        t = torch.randint(0, 1000, (self.batch_size,)).to(self.device)
        out = self.pos_model(t)
        loss = out.mean()
        loss.backward()

        # 1. Access the layer
        first_layer = self.pos_model.mlp[0]

        # 2. Add Type Assertion (Fixes IDE warning + Validates Architecture)
        self.assertIsInstance(first_layer, nn.Linear)

        # 3. Now safely access .weight
        mlp_weight = first_layer.weight

        self.assertIsNotNone(mlp_weight.grad)
        self.assertGreater(mlp_weight.grad.abs().sum().item(), 0.0)

    def test_pos_buffer_persistence(self):
        """[PosEncoding] Frequencies should be buffers, not parameters"""
        self.assertIn('freq', dict(self.pos_model.named_buffers()))
        self.assertNotIn('freq', dict(self.pos_model.named_parameters()))

    # ==========================================
    # TESTS FOR MP_FOURIER
    # ==========================================

    def test_four_output_shape(self):
        """[MP_Fourier] Output shape match"""
        x = torch.randn(self.batch_size).to(self.device)
        out = self.fourier_model(x)
        self.assertEqual(out.shape, (self.batch_size, self.four_channels))

    def test_four_determinism(self):
        """[MP_Fourier] Deterministic behavior (Fixed random basis)"""
        x = torch.randn(self.batch_size).to(self.device)
        out1 = self.fourier_model(x)
        out2 = self.fourier_model(x)
        self.assertTrue(torch.allclose(out1, out2))

    def test_four_gradient_input(self):
        """[MP_Fourier] Differentiable w.r.t input X"""
        x = torch.randn(self.batch_size, device=self.device, requires_grad=True)
        out = self.fourier_model(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_four_dtype_support(self):
        """[MP_Fourier] Supports FP16/Half precision"""
        if self.device.type == 'cpu':
            return  # Skip half test on CPU if not supported

        x = torch.randn(self.batch_size, device=self.device, dtype=torch.float16)
        out = self.fourier_model(x)
        self.assertEqual(out.dtype, torch.float16)

    def test_four_input_constraint(self):
        """[MP_Fourier] Should reject 2D inputs (requires 1D for ger)"""
        x_2d = torch.randn(self.batch_size, 1).to(self.device)
        with self.assertRaises(RuntimeError):
            self.fourier_model(x_2d)


if __name__ == '__main__':
    unittest.main()