import torch
import numpy as np
import unittest
from models import model_internals as m

class TestMagnitudePreservingOps(unittest.TestCase):

    def setUp(self):
        """Set up common parameters for tests."""
        torch.manual_seed(42)
        self.batch_size = 16
        self.seq_len = 32
        self.channels = 64
        self.height = 32
        self.width = 32
        self.emb_dim = 128
        self.time_dim = 64
        # Use a generous tolerance for stochastic variance checks
        self.tolerance = 1e-1
        self.x = torch.randn(2, 3, 16, 16)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def test_mp_silu(self):
        """Test magnitude-preserving SiLU."""
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.assertAlmostEqual(np.sqrt(x.var().item()), 1.0, delta=self.tolerance)

        out = m.mp_silu(x)

        self.assertEqual(x.shape, out.shape)
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_sum(self):
        """Test magnitude-preserving sum (lerp)."""
        a = torch.randn(self.batch_size, self.channels)
        b = torch.randn(self.batch_size, self.channels)
        t = 0.75  # Blending factor

        out = m.mp_sum(a, b, t)

        self.assertEqual(a.shape, out.shape)
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_cat(self):
        """Test magnitude-preserving concatenation."""
        a = torch.randn(self.batch_size, self.channels, self.height, self.width)
        b = torch.randn(self.batch_size, self.channels // 2, self.height, self.width)
        t = 0.5
        dim = 1  # Concatenate on the channel dimension

        out = m.mp_cat(a, b, dim=dim, t=t)

        expected_channels = a.shape[dim] + b.shape[dim]
        self.assertEqual(out.shape, (self.batch_size, expected_channels, self.height, self.width))
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_fourier(self):
        """Test magnitude-preserving Fourier features."""
        num_features = 128
        fourier_layer = m.MP_Fourier(num_channels=num_features)
        x = torch.randn(self.batch_size)  # Input is typically a 1D tensor of timesteps

        out = fourier_layer(x)

        self.assertEqual(out.shape, (self.batch_size, num_features))
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_conv_4d(self):
        """Test magnitude-preserving 2D convolution (4D tensor)."""
        in_channels = self.channels
        out_channels = self.channels * 2
        kernel_size = (3,3)
        conv_layer = m.MP_Conv(in_channels, out_channels, kernel=kernel_size)
        lin_layer = m.MP_Conv(in_channels, out_channels, kernel=())
        conv_layer.train()  # Ensure weights are copied
        lin_layer.train()
        x = torch.randn(self.batch_size, in_channels, self.height, self.width)
        y = torch.randn(self.batch_size, in_channels)
        out = conv_layer(x)
        out2 = lin_layer(y)
        self.assertEqual(out2.shape ,(self.batch_size,out_channels))
        self.assertEqual(out.shape,(self.batch_size, out_channels, self.height, self.width))
        self.assertAlmostEqual(np.sqrt(out.var().item()), np.sqrt(x.var().item()), delta=self.tolerance)

    def test_resample_mode_keep(self):
        """Test that mode='keep' returns the exact same tensor."""
        out = m.resample(self.x, mode='keep')
        self.assertTrue(torch.equal(self.x, out))

    def test_resample_shape_downsample(self):
        """Test that downsampling halves the spatial dimensions."""
        out = m.resample(self.x, mode='down')
        expected_shape = (2, 3, 8, 8)
        self.assertEqual(out.shape, expected_shape)

    def test_resample_shape_upsample(self):
        """Test that upsampling doubles the spatial dimensions."""
        out = m.resample(self.x, mode='up')
        expected_shape = (2, 3, 32, 32)
        self.assertEqual(out.shape, expected_shape)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)