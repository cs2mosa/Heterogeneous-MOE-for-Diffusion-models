import torch
from models import model_components as m
import unittest

class TestUnetBlock(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.emb_dim = 128
        self.img_size = 32
        self.emb = torch.randn(self.batch_size, self.emb_dim)

    def test_encoder_downsample_logic(self):
        in_ch, out_ch = 32, 64
        model = m.Unet_block(in_ch, out_ch, (3, 3), self.emb_dim, resample='down', Type='enc')
        x = torch.randn(self.batch_size, in_ch, 64, 64)

        out = model(x, self.emb)

        # 1. Check Output Shape
        self.assertEqual(out.shape, (self.batch_size, out_ch, 32, 32))

        # 2. Check Logic (FIXED): Access .weights instead of .conv
        # weights shape is (Out, In, H, W). Index 1 is Input Channels.
        # Encoder: x is projected to 'out_ch' BEFORE entering conv_res1.
        self.assertEqual(model.conv_res1.weights.shape[1], out_ch,
                         "Encoder Logic Fail: conv_res1 should accept 'out_channels'")

    def test_decoder_upsample_logic(self):
        in_ch, out_ch = 64, 32
        model = m.Unet_block(in_ch, out_ch, (3, 3), self.emb_dim, resample='up', Type='dec')
        x = torch.randn(self.batch_size, in_ch, 32, 32)

        out = model(x, self.emb)

        # 1. Check Output Shape
        self.assertEqual(out.shape, (self.batch_size, out_ch, 64, 64))

        # 2. Check Logic (FIXED): Access .weights instead of .conv
        # Decoder: x enters conv_res1 with 'in_ch' (raw input).
        self.assertEqual(model.conv_res1.weights.shape[1], in_ch,
                         "Decoder Logic Fail: conv_res1 should accept 'in_channels'")
    def test_identity_channels(self):
        """
        Scenario: In channels == Out channels.
        Ensure conv_skip is None and logic holds.
        """
        in_ch = 32
        model = m.Unet_block(in_ch, in_ch, (3, 3), self.emb_dim, resample='keep', Type='enc')
        x = torch.randn(self.batch_size, in_ch, 32, 32)

        out = model(x, self.emb)

        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(model.conv_skip, "conv_skip should be None if channels match")

    def test_broadcasting_crash(self):
        """
        Scenario: Embedding dimension mismatch.
        Should raise a RuntimeError because Matrix Mult inside MP_Conv or Linear will fail.
        """
        model = m.Unet_block(32, 64, (3, 3), emb_size=128)  # Expects 128
        x = torch.randn(self.batch_size, 32, 32, 32)
        wrong_emb = torch.randn(self.batch_size, 256)  # Passed 256

        with self.assertRaises(RuntimeError):
            model(x, wrong_emb)

    def test_dropout_train_vs_eval(self):
        """
        Scenario: Dropout is active in train, inactive in eval.
        We can't easily check internal values without hooks, but we ensure it runs in both modes.
        """
        model = m.Unet_block(32, 32, (3, 3), self.emb_dim, Dropout=0.5)
        x = torch.randn(self.batch_size, 32, 32, 32)

        # Train Mode
        model.train()
        out_train = model(x, self.emb)
        self.assertEqual(out_train.shape, x.shape)

        # Eval Mode
        model.eval()
        out_eval = model(x, self.emb)
        self.assertEqual(out_eval.shape, x.shape)

    def test_gradients_flow(self):
        """
        Scenario: Ensure the graph is connected and gradients reach the gains.
        """
        model = m.Unet_block(32, 32, (3, 3), self.emb_dim)
        x = torch.randn(self.batch_size, 32, 32, 32, requires_grad=True)

        out = model(x, self.emb)
        loss = out.sum()
        loss.backward()

        # Check if parameters have gradients
        self.assertIsNotNone(model.emb_gain.grad, "emb_gain should have gradients")
        self.assertIsNotNone(model.conv_gain1.grad, "conv_gain1 should have gradients")


if __name__ == '__main__':
    unittest.main()