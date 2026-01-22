import torch
import unittest
import math
from models.Utilities import MaskGenerator

class TestCurriculumMask(unittest.TestCase):
    def setUp(self):
        # Configuration
        self.expert_attrs = [3, 5, 7, 9, 11]  # 5 Experts (e.g. Kernel sizes)
        self.mu = -0.4
        self.std = 1.0
        self.batch_size = 16

        # Instantiate
        self.gen = MaskGenerator(
            expert_attributes=self.expert_attrs,
            p_mean=self.mu,
            p_std=self.std,
            bandwidth=0.2,
            min_active=1
        )

    def test_1_shape(self):
        """Verify output shape is [Batch, Num_Experts]"""
        sigma = torch.tensor([1.0, 0.5, 80.0, 0.002])
        mask = self.gen(sigma)
        self.assertEqual(mask.shape, (4, 5))
        self.assertTrue(torch.all((mask == 0) | (mask == 1)))

    def test_2_safety_valve(self):
        """Verify min_active: every row must have at least N ones"""
        min_active = 2
        gen_safe = MaskGenerator(
            expert_attributes=self.expert_attrs,
            min_active=min_active,
            bandwidth=0.0001  # Extremely narrow, should force safety valve
        )
        sigma = torch.randn(self.batch_size).exp()
        mask = gen_safe(sigma)

        active_counts = mask.sum(dim=-1)
        self.assertTrue(torch.all(active_counts >= min_active))
        print(f"\n[Test 2] Safety Valve check: Min Active {min_active} satisfied.")

    def test_3_high_noise_specialization(self):
        """Verify that VERY high noise activates the largest experts (Attr 11)"""
        # A sigma way above the distribution mean
        high_sigma = torch.tensor([500.0])
        mask = self.gen(high_sigma)

        # Expert index 4 is Attr 11 (the largest)
        self.assertEqual(mask[0, 4].item(), 1.0)
        # Smallest expert (index 0) should likely be masked out
        self.assertEqual(mask[0, 0].item(), 0.0)
        print("[Test 3] High noise correctly targets large-kernel experts.")

    def test_4_low_noise_specialization(self):
        """Verify that VERY low noise activates the smallest experts (Attr 3)"""
        low_sigma = torch.tensor([0.0001])
        mask = self.gen(low_sigma)

        # Expert index 0 is Attr 3 (the smallest)
        self.assertEqual(mask[0, 0].item(), 1.0)
        # Largest expert (index 4) should be masked out
        self.assertEqual(mask[0, 4].item(), 0.0)
        print("[Test 4] Low noise correctly targets small-kernel experts.")

    def test_5_median_logic(self):
        """
        Verify the CDF logic: the median of the Log-Normal dist (exp(mu))
        should result in a percentile of exactly 0.5.
        """
        median_sigma = torch.tensor([math.exp(self.mu)])

        # We manually calculate where the mask should be centered
        # 0.5 percentile should be the middle of [0, 1] scales
        mask = self.gen(median_sigma)

        # In attributes [3, 5, 7, 9, 11], 7 is exactly the center (0.5)
        # So expert index 2 (Attr 7) must be active
        self.assertEqual(mask[0, 2].item(), 1.0)
        print(f"[Test 5] Median noise {median_sigma.item():.4f} correctly targets center expert.")

    def test_6_bandwidth_expansion(self):
        """Verify that increasing bandwidth activates more experts"""
        sigma = torch.tensor([1.0])

        gen_narrow = MaskGenerator(self.expert_attrs, bandwidth=0.01)
        gen_wide = MaskGenerator(self.expert_attrs, bandwidth=0.8)

        mask_n = gen_narrow(sigma)
        mask_w = gen_wide(sigma)

        self.assertGreater(mask_w.sum(), mask_n.sum())
        print(f"[Test 6] Bandwidth test: Wide ({mask_w.sum()}) > Narrow ({mask_n.sum()}).")

    def test_7_no_grad(self):
        """Verify that the operation is truly non-differentiable (saves memory)"""
        sigma = torch.tensor([1.0], requires_grad=True)
        mask = self.gen(sigma)
        self.assertFalse(mask.requires_grad)


if __name__ == '__main__':
    unittest.main()