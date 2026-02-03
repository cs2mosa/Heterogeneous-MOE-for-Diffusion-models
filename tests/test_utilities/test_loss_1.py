import torch
import torch.nn.functional as F
import unittest


class TestEDMLoss(unittest.TestCase):
    def setUp(self):
        self.B, self.C, self.H, self.W = 4, 3, 16, 16
        self.num_experts = 5
        self.sigma_data = 0.5

        self.criterion = Utilities.EDM_LOSS(num_experts=self.num_experts, sigma_data=self.sigma_data)

    def test_1_output_scalar(self):
        """Test: Is the total loss a scalar? (Prevents broadcasting bugs)"""
        sigma = torch.rand(self.B, 1, 1, 1)
        x = torch.randn(self.B, self.C, self.H, self.W)

        # Mocking model output tuple
        denoised = torch.randn_like(x)
        u_probs = F.softmax(torch.randn(self.B, self.num_experts), dim=-1)
        u_logits = torch.randn(self.B, self.num_experts)
        v_probs = F.softmax(torch.randn(self.B, self.num_experts), dim=-1)
        v_logits = torch.randn(self.B, self.num_experts)
        s_probs = F.softmax(torch.randn(self.B, 2), dim=-1)
        logvar = torch.randn(self.B, 1, 1, 1)

        out_model = {"denoised":denoised,
                    "Unet_router_loss":u_probs,
                    "Unet_raw" :u_logits,
                    "vit_router_loss": v_probs,
                    "vit_raw":v_logits,
                    "scaling_net_out": s_probs,
                    "log_var":logvar}

        loss_dict = self.criterion(sigma, x, out_model)

        self.assertIsInstance(loss_dict['loss'], torch.Tensor)
        self.assertEqual(loss_dict['loss'].ndim, 0)  # Scalar check
        print("\n[Test 1 Passed] Loss output is a scalar.")

    def test_2_gradient_flow(self):
        """Test: Do all components contribute to the gradient?"""
        sigma = torch.rand(self.B, 1, 1, 1)
        x = torch.randn(self.B, self.C, self.H, self.W)

        # Create tensors with requires_grad=True
        denoised = torch.randn_like(x, requires_grad=True)
        u_logits = torch.randn(self.B, self.num_experts, requires_grad=True)
        v_logits = torch.randn(self.B, self.num_experts, requires_grad=True)
        s_logits = torch.randn(self.B, 2, requires_grad=True)
        logvar = torch.randn(self.B, 1, 1, 1, requires_grad=True)

        # Derive probs from logits so we can test grad through softmax
        u_probs = F.softmax(u_logits, dim=-1)
        v_probs = F.softmax(v_logits, dim=-1)
        s_probs = F.softmax(s_logits, dim=-1)

        out_model = {"denoised": denoised,
                     "Unet_router_loss": u_probs,
                     "Unet_raw": u_logits,
                     "vit_router_loss": v_probs,
                     "vit_raw": v_logits,
                     "scaling_net_out": s_probs,
                     "log_var": logvar}

        loss_dict = self.criterion(sigma, x, out_model)
        loss_dict['loss'].backward()

        self.assertIsNotNone(denoised.grad, "No grad to Expert output")
        self.assertIsNotNone(u_logits.grad, "No grad to Unet Router")
        self.assertIsNotNone(s_logits.grad, "No grad to Scaling Router")
        self.assertIsNotNone(logvar.grad, "No grad to EDM2 Log-variance")
        print("[Test 2 Passed] Gradients flow through all model outputs.")

    def test_3_balance_penalty(self):
        """Test: Does imbalanced routing result in higher loss than balanced?"""
        # Balanced Case: All experts get 1/N probability
        balanced_probs = torch.ones(self.B, self.num_experts) / self.num_experts
        loss_balanced = Utilities.load_balance(balanced_probs, self.num_experts)

        # Imbalanced Case: All batch items go to Expert 0
        imbalanced_probs = torch.zeros(self.B, self.num_experts)
        imbalanced_probs[:, 0] = 1.0
        loss_imbalanced = Utilities.load_balance(imbalanced_probs, self.num_experts)

        self.assertGreater(loss_imbalanced, loss_balanced)
        # Mathematical invariant: Balanced should equal 1.0 (N * sum((1/N)^2))
        self.assertAlmostEqual(loss_balanced.item(), 1.0, places=5)
        print(
            f"[Test 3 Passed] Imbalance penalty working. Balanced: {loss_balanced.item():.2f}, Imbalanced: {loss_imbalanced.item():.2f}")

    def test_4_entropy_behavior(self):
        """Test: Does uniform scaling have higher entropy (lower loss) than one-hot scaling?"""
        # High Entropy (0.5, 0.5)
        high_ent_probs = torch.tensor([[0.5, 0.5]])
        val_high = Utilities.entropy_loss(high_ent_probs)

        # Low Entropy (0.99, 0.01)
        low_ent_probs = torch.tensor([[0.99, 0.01]])
        val_low = Utilities.entropy_loss(low_ent_probs)

        self.assertGreater(val_high, val_low)
        print(f"[Test 4 Passed] Entropy working. High: {val_high.item():.4f}, Low: {val_low.item():.4f}")

    def test_5_z_loss_penalty(self):
        """Test: Do large logits increase the Z-loss?"""
        small_logits = torch.tensor([[1.0, 1.0, 1.0]])
        large_logits = torch.tensor([[10.0, 10.0, 10.0]])

        loss_small = Utilities.z_loss(small_logits)
        loss_large = Utilities.z_loss(large_logits)

        self.assertGreater(loss_large, loss_small)
        print(f"[Test 5 Passed] Z-loss correctly penalizes large logits.")


if __name__ == '__main__':
    unittest.main()