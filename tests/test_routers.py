import torch
import models.model_components as m
import unittest

class TestScalingRouter(unittest.TestCase):
    def setUp(self):
        """Set up a scaling router and dummy data for all tests."""
        self.batch_size = 8
        self.in_channels = 3
        self.image_dims = (32, 32)
        self.num_paths = 2

        self.router = m.Scaling_router(in_channels=self.in_channels, num_experts=self.num_paths)
        self.dummy_input = torch.randn(self.batch_size, self.in_channels, *self.image_dims)

    def test_output_shape(self):
        """Checks if the output tensor has the correct shape [batch_size, num_paths]."""
        self.router.eval()
        output = self.router(self.dummy_input)
        self.assertEqual(output.shape, (self.batch_size, self.num_paths))

    def test_scaling_factor_properties(self):
        """Checks if scaling factors are positive and sum to 2.0 for each batch item."""
        self.router.eval()
        output = self.router(self.dummy_input)
        # Check that all factors are positive
        self.assertTrue(torch.all(output >= 0))
        # Check that the sum of factors for each image is 2.0
        sum_of_factors = output.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_of_factors, torch.tensor(2.0)))

    def test_eval_mode_is_deterministic(self):
        """Checks if two consecutive forward passes in eval mode produce identical results."""
        self.router.eval()
        output1 = self.router(self.dummy_input)
        output2 = self.router(self.dummy_input)
        self.assertTrue(torch.equal(output1, output2))

    def test_train_mode_is_stochastic(self):
        """Checks if two consecutive forward passes in train mode produce different results."""
        self.router.train()
        output1 = self.router(self.dummy_input)
        output2 = self.router(self.dummy_input)
        # Due to dropout and noise, outputs should be different
        self.assertFalse(torch.equal(output1, output2))


class TestHardRouter(unittest.TestCase):
    def setUp(self):
        """Set up hard routers and dummy data for all tests."""
        self.batch_size = 16
        self.in_channels = 3
        self.image_dims = (32, 32)
        self.num_experts = 5
        self.channels = 3  # in_channels
        self.height = 32  # image_dims[0]
        self.width = 32
        self.top_k = 1
        self.router = m.Router(in_channels=self.channels,
                             top_k=self.top_k,
                             num_experts=self.num_experts)
        # Create routers with different top_k values to test parameterization
        self.router_k1 = m.Router(in_channels=self.in_channels, num_experts=self.num_experts, top_k=1)
        self.router_k2 = m.Router(in_channels=self.in_channels, num_experts=self.num_experts, top_k=2)
        self.dummy_input = torch.randn(self.batch_size, self.in_channels, *self.image_dims)

    def test_output_shapes(self):
        """Checks if the two output tensors have the correct shape [batch_size, num_experts]."""
        self.router_k1.eval()
        sparse_weights, gate_probs = self.router_k1(self.dummy_input)
        expected_shape = (self.batch_size, self.num_experts)
        self.assertEqual(sparse_weights.shape, expected_shape)
        self.assertEqual(gate_probs.shape, expected_shape)

    def test_gate_probs_is_valid_distribution(self):
        """Checks if the dense gate_probs tensor is a valid probability distribution (sums to 1)."""
        self.router_k1.eval()
        _, gate_probs = self.router_k1(self.dummy_input)
        sum_of_probs = gate_probs.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_of_probs, torch.tensor(1.0)))

    def test_sparsity_and_sum_top_k_1(self):
        """Checks for k=1 that exactly one expert is chosen and its weight is 1.0."""
        self.router_k1.eval()
        sparse_weights, _ = self.router_k1(self.dummy_input)

        # Check that each row has exactly k=1 non-zero elements
        non_zero_counts = torch.count_nonzero(sparse_weights, dim=-1)
        self.assertTrue(torch.all(non_zero_counts == 1))

        # Check that the sum of weights for each item is 1.0
        sum_of_weights = sparse_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_of_weights, torch.tensor(1.0)))

    def test_sparsity_and_sum_top_k_2(self):
        """Checks for k=2 that exactly two experts are chosen and their weights sum to 1.0."""
        self.router_k2.eval()
        sparse_weights, _ = self.router_k2(self.dummy_input)

        # Check that each row has exactly k=2 non-zero elements
        non_zero_counts = torch.count_nonzero(sparse_weights, dim=-1)
        self.assertTrue(torch.all(non_zero_counts == 2))

        # Check that the sum of weights for each item is 1.0
        sum_of_weights = sparse_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sum_of_weights, torch.tensor(1.0)))

    def test_eval_mode_is_deterministic(self):
        """Checks if two consecutive forward passes in eval mode produce identical results."""
        self.router_k1.eval()
        sparse1, probs1 = self.router_k1(self.dummy_input)
        sparse2, probs2 = self.router_k1(self.dummy_input)
        self.assertTrue(torch.equal(sparse1, sparse2))
        self.assertTrue(torch.equal(probs1, probs2))

    def test_mask_enforcement_output(self):
        """ Test 1: Functionality - Are weights strictly zero where masked? """
        # Generate input with (B, 3, 32, 32)
        x = torch.randn(self.batch_size, self.channels, self.image_dims[0], self.image_dims[1])

        # Create a mask: (Batch, Num_Experts) -> (16, 5)
        # Default: All experts enabled
        mask = torch.ones(self.batch_size, self.num_experts)

        # TEST LOGIC:
        # 1. Mask Expert Index 3 GLOBALLY (for all batches)
        # Valid because num_experts=5 (Indices: 0,1,2,3,4)
        mask[:, 3] = 0

        # 2. Mask Expert Index 0 specifically for Batch Index 0
        mask[0, 0] = 0

        weights, _ = self.router_k1(x, mask=mask)

        # Check Batch 0, Expert 0 -> Should be 0.0
        self.assertEqual(weights[0, 0].item(), 0.0, "Masked expert 0 received non-zero weight!")

        # Check All Batches, Expert 3 -> Should be 0.0
        self.assertTrue(torch.all(torch.eq(weights[:, 3], 0.0)).item(), "Globally masked expert received weights!")

        # Check that we still selected Top-K experts (weights sum to 1.0)
        sums = weights.sum(dim=1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums)), "Weights do not sum to 1 after masking")

        # Sanity check: ensure at least one weight is non-zero (since k=2)
        self.assertNotEqual(weights.sum().item(), 0.0)

    def test_gradient_blocking(self):
        """
        Test 2: The Critical Test - Do gradients stop flowing to masked experts?
        """
        x = torch.randn(self.batch_size, self.channels, self.image_dims[0], self.image_dims[1])

        # Define the mask: Block Expert 0 entirely
        mask = torch.ones(self.batch_size, self.num_experts)
        mask[:, 0] = 0  # Expert 0 is BANNED

        # --- Simulate The Experts ---
        # 5 Experts, returning vectors of size 64
        expert_outputs = torch.randn(self.batch_size, self.num_experts, 64, requires_grad=True)

        # Forward pass Router
        weights, _ = self.router_k1(x, mask=mask)

        # Forward pass "Mixture of Experts" simulation
        # Broadcast weights: (16, 5) -> (16, 5, 1)
        weighted_output = (weights.unsqueeze(-1) * expert_outputs).sum()

        # Backward
        weighted_output.backward()

        # --- ASSERTIONS ---

        # 1. Expert 0 (Masked) should have ZERO gradient
        grad_exp0 = expert_outputs.grad[:, 0, :].abs().sum().item()
        self.assertEqual(grad_exp0, 0.0, f"Masked expert received gradient flow! ({grad_exp0})")

        # 2. Check unmasked experts (Indices 1,2,3,4)
        # Since K=2 ,and we have 4 valid experts, some will get gradients.
        grad_valid_exps = expert_outputs.grad[:, 1:, :].abs().sum().item()
        self.assertGreater(grad_valid_exps, 0.0, "Unmasked experts received no gradients!")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)