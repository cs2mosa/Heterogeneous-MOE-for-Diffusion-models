import torch
from models.model_config1 import preconditioned_HDMOEM
import unittest


class TestPreconditionedHDMOEM(unittest.TestCase):

    def setUp(self):
        """
        Set up the common hyperparameters and input shapes for testing.
        """
        self.batch_size = 2
        self.in_channels = 4
        self.img_res = 32
        self.time_dim = 64
        self.text_dim = 128
        self.num_experts = 2

        # Configuration dictionary matching __init__ arguments
        self.config = {
            'IN_in_channels': self.in_channels,
            'IN_img_resolution': self.img_res,
            'time_emb_dim': self.time_dim,
            'text_emb_dim': self.text_dim,
            'num_experts': self.num_experts,
            'top_k': 1,
            'Fourier_bandwidth': 1.0,
            'VIT_num_blocks': 1,
            'VIT_patch_sizes': [4, 4],
            'VIT_num_groups': 2,
            'VIT_num_heads': 2,
            'VIT_emb_size': 32,
            'Unet_num_blocks': 1,
            'Unet_channel_mult': [1, 2],
            'Unet_kernel_sizes': [(3, 3), (3, 3)],
            'Unet_model_channels': 32,
            'Unet_channel_mult_emb': 2,
            'Unet_label_balance': 0.5,
            'Unet_concat_balance': 0.5,
            'sigma_data': 0.5,
            'log_var_channels': 32
        }

    def test_01_forward_shapes_basic(self):
        """
        Test Case 1: Standard Forward Pass
        Verify that output dimensions match input dimensions (Denoising property).
        """
        model = preconditioned_HDMOEM(**self.config)

        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res)
        sigma = torch.rand(self.batch_size)  # (Batch,)
        text_emb = torch.randn(self.batch_size, self.text_dim)  # (Batch, Dim)

        # Masks (Batch, Num_Experts)
        unet_mask = torch.ones(self.batch_size, self.num_experts)
        vit_mask = torch.ones(self.batch_size, self.num_experts)
        zeta = 0.1

        output, unet_gate, vit_gate, log_var = model(
            x, sigma, text_emb, unet_mask, vit_mask, zeta, return_log_var=False
        )

        # Assertions
        self.assertEqual(output.shape, x.shape, "Output image shape mismatch")
        self.assertEqual(unet_gate.shape, (self.batch_size, self.num_experts), "UNet gate shape mismatch")
        self.assertEqual(vit_gate.shape, (self.batch_size, self.num_experts), "ViT gate shape mismatch")
        self.assertIsNone(log_var, "Log var should be None when return_log_var=False")

    def test_02_return_log_var(self):
        """
        Test Case 2: Uncertainty Estimation
        Verify that setting return_log_var=True returns a valid tensor.
        """
        model = preconditioned_HDMOEM(**self.config)

        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res)
        sigma = torch.rand(self.batch_size)
        text_emb = torch.randn(self.batch_size, self.text_dim)
        unet_mask = torch.ones(self.batch_size, self.num_experts)
        vit_mask = torch.ones(self.batch_size, self.num_experts)

        _, _, _, log_var = model(
            x, sigma, text_emb, unet_mask, vit_mask, zeta=0.1, return_log_var=True
        )

        self.assertIsNotNone(log_var)
        # Assuming log_var output is broadcastable per image, e.g., (Batch, 1, 1, 1)
        self.assertEqual(log_var.shape[0], self.batch_size)
        self.assertTrue(log_var.requires_grad, "Log var should track gradients")

    def test_03_text_embedding_sequence(self):
        """
        Test Case 3: Text Embedding Dimensions
        The code handles text_emb.mean(dim=1) if ndim=3.
        Verify passing (Batch, Seq_Len, Dim) works.
        """
        model = preconditioned_HDMOEM(**self.config)
        seq_len = 77
        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res)
        sigma = torch.rand(self.batch_size)
        # Input shape (Batch, Seq_Len, Dim)
        text_emb_seq = torch.randn(self.batch_size, seq_len, self.text_dim)

        unet_mask = torch.ones(self.batch_size, self.num_experts)
        vit_mask = torch.ones(self.batch_size, self.num_experts)

        output, _, _, _ = model(
            x, sigma, text_emb_seq, unet_mask, vit_mask, zeta=0.1
        )
        self.assertEqual(output.shape, x.shape)

    def test_04_gradient_propagation(self):
        """
        Test Case 4: Training Stability
        Verify that gradients flow all the way back to the input and parameters.
        """
        model = preconditioned_HDMOEM(**self.config)

        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res, requires_grad=True)
        sigma = torch.rand(self.batch_size, requires_grad=True)
        text_emb = torch.randn(self.batch_size, self.text_dim)
        unet_mask = torch.ones(self.batch_size, self.num_experts)
        vit_mask = torch.ones(self.batch_size, self.num_experts)

        output, u_probs, v_probs, _ = model(x, sigma, text_emb, unet_mask, vit_mask, zeta=0.1)

        # Simulating a loss
        loss = output.mean() + u_probs.mean() + v_probs.mean()
        loss.backward()

        # Check gradients
        self.assertIsNotNone(x.grad, "Input x did not receive gradients")
        self.assertIsNotNone(sigma.grad, "Sigma did not receive gradients")

        # Check if internal parameters have grads (e.g., Fourier layers)
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad, "Model parameters did not receive gradients")

    def test_05_router_masking_edge_case(self):
        """
        Test Case 5: Masking Logic
        Verify behavior when some experts are masked out (set to 0).
        """
        model = preconditioned_HDMOEM(**self.config)

        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res)
        sigma = torch.rand(self.batch_size)
        text_emb = torch.randn(self.batch_size, self.text_dim)

        # Only activate the first expert
        unet_mask = torch.zeros(self.batch_size, self.num_experts)
        unet_mask[:, 0] = 1.0
        vit_mask = torch.zeros(self.batch_size, self.num_experts)
        vit_mask[:, 0] = 1.0

        output, _, _, _ = model(x, sigma, text_emb, unet_mask, vit_mask, zeta=0.1)

        # Ensure it didn't crash and output is valid
        self.assertEqual(output.shape, x.shape)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaNs with partial masking")

    def test_06_sparse_router_empty_mask(self):
        """
        Test Case 6: Empty Mask (No experts active)
        This checks if the code handles the case where mask is all zeros safely.
        """
        model = preconditioned_HDMOEM(**self.config)


        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res)
        sigma = torch.rand(self.batch_size)
        text_emb = torch.randn(self.batch_size, self.text_dim)

        # All experts inactive
        unet_mask = torch.zeros(self.batch_size, self.num_experts)
        vit_mask = torch.zeros(self.batch_size, self.num_experts)

        # The helper function `router_to_unet_experts` has a check `if not mask.any(): continue`.
        # If all masks are 0, output should remain zeros (from initialization `output = torch.zeros_like(x)`).
        # But `HDMOEM` logic adds `out_gated_attn = Wx * out_Unet_expert...`
        # If expert output is 0, the final output should theoretically be influenced only by bias or be zero/residual.

        output, _, _, _ = model(x, sigma, text_emb, unet_mask, vit_mask, zeta=0.1)

        self.assertEqual(output.shape, x.shape)
        # Since mocks return random data, logic implies if experts are skipped, result might be specific.
        # We just verify it runs without index errors.

    def test_07_device_movement(self):
        """
        Test Case 7: GPU Compatibility
        Verify the model moves to GPU and runs if available.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        device = torch.device("cuda")
        model = preconditioned_HDMOEM(**self.config).to(device)


        x = torch.randn(self.batch_size, self.in_channels, self.img_res, self.img_res).to(device)
        sigma = torch.rand(self.batch_size).to(device)
        text_emb = torch.randn(self.batch_size, self.text_dim).to(device)
        unet_mask = torch.ones(self.batch_size, self.num_experts).to(device)
        vit_mask = torch.ones(self.batch_size, self.num_experts).to(device)

        output, _, _, _ = model(x, sigma, text_emb, unet_mask, vit_mask, zeta=0.1)

        self.assertEqual(output.device.type, 'cuda')

    def test_08_preconditioning_math_safety(self):
        """
        Test Case 8: Preconditioning Logic Checks
        Ensure c_skip, c_in, c_out don't produce NaNs when sigma is 0 or very large.
        """
        model = preconditioned_HDMOEM(**self.config)


        x = torch.randn(2, self.in_channels, self.img_res, self.img_res)
        text_emb = torch.randn(2, self.text_dim)
        unet_mask = torch.ones(2, self.num_experts)
        vit_mask = torch.ones(2, self.num_experts)

        # Case A: Sigma = 0 (Clean image)
        sigma_zero = torch.zeros(2)
        out_zero, _, _, _ = model(x, sigma_zero, text_emb, unet_mask, vit_mask, zeta=0.1)
        self.assertFalse(torch.isnan(out_zero).any(), "NaN detected with sigma=0")

        # Case B: Sigma = Large
        sigma_large = torch.tensor([1000.0, 1000.0])
        out_large, _, _, _ = model(x, sigma_large, text_emb, unet_mask, vit_mask, zeta=0.1)
        self.assertFalse(torch.isnan(out_large).any(), "NaN detected with large sigma")


if __name__ == '__main__':
    unittest.main()