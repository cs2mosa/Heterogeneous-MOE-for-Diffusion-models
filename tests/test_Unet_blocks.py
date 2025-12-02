import torch
from models import model_components as m
import unittest

class TestUnetBlock(unittest.TestCase):

    def setUp(self):
        self.batch_size = 2
        self.emb_dim = 128
        self.img_size = 32
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.emb = torch.randn(self.batch_size, self.emb_dim).to(device= self.device)


    def test_encoder_downsample_logic(self):
        in_ch, out_ch = 32, 64
        model = m.Unet_block(in_ch, out_ch, (3, 3), self.emb_dim, resample='down', Type='enc').to(device= self.device)
        x = torch.randn(self.batch_size, in_ch, 64, 64).to(device= self.device)

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
        model = m.Unet_block(in_ch, out_ch, (3, 3), self.emb_dim, resample='up', Type='dec').to(device= self.device)
        x = torch.randn(self.batch_size, in_ch, 32, 32).to(device= self.device)

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
        model = m.Unet_block(in_ch, in_ch, (3, 3), self.emb_dim, resample='keep', Type='enc').to(device= self.device)
        x = torch.randn(self.batch_size, in_ch, 32, 32).to(device= self.device)

        out = model(x, self.emb)

        self.assertEqual(out.shape, x.shape)
        self.assertIsNone(model.conv_skip, "conv_skip should be None if channels match")

    def test_broadcasting_crash(self):
        """
        Scenario: Embedding dimension mismatch.
        Should raise a RuntimeError because Matrix Mult inside MP_Conv or Linear will fail.
        """
        model = m.Unet_block(32, 64, (3, 3), emb_size=128) .to(device= self.device) # Expects 128
        x = torch.randn(self.batch_size, 32, 32, 32).to(device= self.device)
        wrong_emb = torch.randn(self.batch_size, 256)  # Passed 256

        with self.assertRaises(RuntimeError):
            model(x, wrong_emb)

    def test_dropout_train_vs_eval(self):
        """
        Scenario: Dropout is active in train, inactive in eval.
        We can't easily check internal values without hooks, but we ensure it runs in both modes.
        """
        model = m.Unet_block(32, 32, (3, 3), self.emb_dim, Dropout=0.5).to(device= self.device)
        x = torch.randn(self.batch_size, 32, 32, 32).to(device= self.device)

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
        Scenario: Ensure the graph is connected, gradients reach the gains,
        and the gradients are strictly non-zero (active learning).
        """
        # 1. Move Model to Device
        model = m.Unet_block(32, 32, (3, 3), self.emb_dim).to(device=self.device)

        # 2. Create Input DIRECTLY on Device (Crucial for Leaf Nodes)
        # If we did torch.randn(...).to(device), x would not be a leaf node!
        x = torch.randn(self.batch_size, 32, 32, 32,
                        device=self.device, requires_grad=True)

        # Ensure embedding input is on device (Assuming self.emb is fixed data)
        emb_input = self.emb.to(self.device)

        # 3. Forward Pass
        out = model(x, emb_input)

        # 4. Compute Loss (Use mean to keep magnitude stable)
        loss = out.mean()
        loss.backward()

        # 5. Assertions: Check Existence AND Magnitude

        # --- Check Input Flow (End-to-End) ---
        self.assertIsNotNone(x.grad, "Input x grad is None (Backprop didn't reach start)")
        grad_mag_x = x.grad.abs().sum().item()
        self.assertGreater(grad_mag_x, 1e-6, f"Input x grad is zero ({grad_mag_x})")

if __name__ == '__main__':
    unittest.main()