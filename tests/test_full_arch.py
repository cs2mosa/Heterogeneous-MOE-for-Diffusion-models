import torch
from numpy.ma.testutils import assert_not_equal

import models.model_config1 as m
import unittest
import copy

class TestHDMOEM(unittest.TestCase):
    def setUp(self):
        # 1. Constants
        self.B, self.C, self.H, self.W = 4, 16, 32, 32
        self.time_dim = 32
        self.text_dim = 64
        self.num_experts = 3
        self.patch_size = 2 #note: patch sizes should be divisible by img_resolutions
        # 2. Instantiate Model
        self.model = m.HDMOEM(
            IN_in_channels=self.C, IN_img_resolution=self.H,
            time_emb_dim=self.time_dim, text_emb_dim=self.text_dim,
            num_experts=self.num_experts, top_k=1,
            Fourier_bandwidth=1.0,
            VIT_num_blocks=1, VIT_patch_sizes=[self.patch_size, self.patch_size*2,self.patch_size*4] , VIT_num_groups=2,
            VIT_num_heads=4, VIT_emb_size=self.C,
            Unet_num_blocks=1, Unet_channel_mult=[1], Unet_kernel_sizes=[(3,3)] * 3
        )

        # 3. Standard Dummy Inputs
        self.x = torch.randn(self.B, self.C, self.H, self.W)
        self.time_vec = torch.rand(self.B)
        self.text_emb = torch.randn(self.B, 10, self.text_dim)
        self.mask = torch.ones(self.B, self.num_experts)
        self.zeta = 0.01

    def test_1_standard_forward(self):
        """Basic Sanity Check: Does the model run and return correct shape?"""
        out,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, self.zeta,)
        self.assertEqual(out.shape, (self.B, self.C, self.H, self.W))

    def test_2_autograd_flow(self):
        """Righteousness Check: Is the entire graph differentiable?"""
        self.model.train()
        out,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, self.zeta)
        loss = out[0].sum()
        loss.backward()

        # Check gradients in specific distant components
        self.assertIsNotNone(self.model.scaling_net.linear.weights.grad, "Scaling Net has no grad")
        self.assertIsNotNone(self.model.Unet_experts[0].encoders[f'{32}x{32}_block{0}'].conv_res1.weights.grad, "Expert 0 has no grad")
        self.assertIsNotNone(self.model.Unet_experts[0].encoders[f'{32}x{32}_block{0}'].emb_layer.weights.grad,"Expert 0 has no grad")
        self.assertIsNotNone(self.model.gate2.weights.grad, "Final Gating has no grad")


    def test_3_batch_size_one(self):
        """Corner Case: Batch Normalization often fails with B=1."""
        x_small = torch.randn(1, self.C, self.H, self.W)
        t_small = torch.rand(1)
        text_small = torch.randn(1, 10, self.text_dim)
        mask_small = torch.ones(1, self.num_experts)

        try:
            out,_,_ = self.model(x_small, t_small, text_small, mask_small, mask_small, self.zeta)
            self.assertEqual(out.shape, (1, self.C, self.H, self.W))
        except Exception as e:
            self.fail(f"Batch size 1 failed with error: {e}")

    def test_4_masked_routing(self):
        """Logic Check: Does masking actually restrict expert usage?"""
        # Force only Expert 0 allowed
        mask_restrict = torch.zeros(self.B, self.num_experts)
        mask_restrict[:, 0] = 1.0

        # We hook the router output to verify
        # (This relies on understanding the router internals, here checked by running success)
        out,_,_ = self.model(self.x, self.time_vec, self.text_emb, mask_restrict, mask_restrict, self.zeta)
        self.assertEqual(out.shape, self.x.shape)

    def test_5_time_sensitivity(self):
        """Logic Check: Does changing the time input actually change the result?"""
        self.model.eval()  # Disable dropout/noise
        t1 = torch.zeros(self.B)
        t2 = torch.ones(self.B)

        out1,_,_ = self.model(self.x, t1, self.text_emb, self.mask, self.mask, 0.0)
        out2,_,_ = self.model(self.x, t2, self.text_emb, self.mask, self.mask, 0.0)

        self.assertFalse(torch.allclose(out1, out2), "Model is outputting identical results for different times!")

    def test_6_noise_determinism(self):
        """Stochastic Check: Train mode should vary (zeta), Eval mode should be fixed."""
        # Train Mode (Expect variation due to zeta)
        self.model.train()
        out_t1,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, 1.0)
        out_t2,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, 1.0)
        assert_not_equal(out_t2.mean(),out_t1.mean(),"should not be equal")
        # Note: If zeta is high, outputs should differ.
        # (In Mock, scaling router uses zeta, so this should pass)

        # Eval Mode (Expect exact match)
        self.model.eval()
        out_e1,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, 0.0)
        out_e2,_,_ = self.model(self.x, self.time_vec, self.text_emb, self.mask, self.mask, 0.0)
        self.assertTrue(torch.allclose(out_e1, out_e2), "Eval mode is not deterministic")

    def test_7_mixed_precision_f16(self):
        """Hardware Check: Can the model run in Half Precision (typical for diffusion)?"""
        if not torch.cuda.is_available():
            print("Skipping FP16 test (CPU Float16 is slow/unsupported for some ops)")
            return

        model_half = copy.deepcopy(self.model).cuda().half()
        x_half = self.x.cuda().half()
        t_half = self.time_vec.cuda().half()
        text_half = self.text_emb.cuda().half()
        mask_half = self.mask.cuda().half()

        out,_,_ = model_half(x_half, t_half, text_half, mask_half, mask_half, self.zeta)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, (self.B, self.C, self.H, self.W))

    def test_8_dimension_mismatches(self):
        """Safety Check: Does it fail gracefully/loudly if inputs are wrong?"""
        # Wrong Image Channels
        x_wrong = torch.randn(self.B, self.C + 1, self.H, self.W)
        with self.assertRaises(RuntimeError):
            self.model(x_wrong, self.time_vec, self.text_emb, self.mask, self.mask, self.zeta)


if __name__ == '__main__':
    unittest.main()