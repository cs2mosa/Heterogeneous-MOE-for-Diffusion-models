import torch
import models.model_components as m
import unittest

class TestDiffiTViTBlock(unittest.TestCase):

    def setUp(self):
        # Common configuration
        self.batch_size = 4
        self.seq_len = 256  # 16x16 patch grid
        self.emb_dim = 64
        self.num_groups = 8  # Must be divisor of emb_dim
        self.num_heads = 4

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.block = m.Vit_block(
            num_heads=self.num_heads,
            num_groups=self.num_groups,
            num_channels=self.emb_dim,  # Isotropic: In == Emb
            seq_ln=self.seq_len,
            emb_dim=self.emb_dim,
            time_dim=self.emb_dim
        ).to(self.device)

    def test_forward_shape_match(self):
        """ Test 1: Does output shape match input shape? """
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        t = torch.randn(self.batch_size, self.emb_dim).to(self.device)

        out = self.block(x, time_embedding=t)

        self.assertEqual(out.shape, x.shape,
                         f"Shape Mismatch! Input: {x.shape}, Output: {out.shape}")

    def test_forward_without_time(self):
        """ Test 2: Does it work without optional time embedding? """
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim).to(self.device)
        out = self.block(x, time_embedding=None)
        self.assertEqual(out.shape, x.shape)

    def test_gradient_flow(self):
        """ Test 3: Do gradients propagate back to input and weights? """
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim, requires_grad=True, device=self.device)
        t = torch.randn(self.batch_size, self.emb_dim, requires_grad=True,device=self.device)

        out = self.block(x, t)
        loss = out.mean()
        loss.backward()

        self.assertIsNotNone(x.grad, "Input gradient is None!")
        self.assertIsNotNone(t.grad, "Time gradient is None!")
        self.assertIsNotNone(self.block.linear1.weights.grad, "Weights did not update!")

    def test_variable_sequence_length(self):
        """ Test 4: Can the block handle different sequence lengths? """
        # e.g., if we run inference on a different aspect ratio or resolution
        seq_len_new = 100  # Arbitrary number
        x = torch.randn(self.batch_size, seq_len_new, self.emb_dim).to(self.device)
        t = torch.randn(self.batch_size, self.emb_dim).to(self.device)

        out = self.block(x, t)
        self.assertEqual(out.shape[1], seq_len_new)

    def test_batch_size_one(self):
        """ Test 5: Does GroupNorm/Reshape explode on Batch=1? """
        x = torch.randn(1, self.seq_len, self.emb_dim).to(self.device)
        t = torch.randn(1, self.emb_dim).to(self.device)

        out = self.block(x, t)
        self.assertEqual(out.shape, (1, self.seq_len, self.emb_dim))

    def test_channel_mismatch_fallback(self):
        """ Test 6: (Corner Case) What if someone tries to use it for projection? """
        input_dim = 32
        block_mismatch = m.Vit_block(
            num_heads=self.num_heads,
            num_groups=4,  # 32 is divisible by 4
            num_channels=input_dim,
            seq_ln=self.seq_len,
            emb_dim=self.emb_dim,  # 64
        ).to(self.device)

        x = torch.randn(self.batch_size, self.seq_len, input_dim).to(self.device)

        # Should return transformed shape (Batch, Seq, Emb_Dim)
        # NOT Input shape, and should NOT crash on residual addition
        out = block_mismatch(x)
        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.emb_dim))

    def test_cpu_to_gpu_migration_flow(self):
        """
        Test 7: Verify gradients flow when inputs are created on CPU
        and migrated to GPU (Standard DataLoder scenario).
        """
        if not torch.cuda.is_available():
            print("Skipping Migration test (No GPU)")
            return

        device = torch.device('cuda')
        model = self.block.to(device)

        # 1. Create Input on CPU (Simulating a DataLoader)
        x_cpu = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        t_cpu = torch.randn(self.batch_size, self.emb_dim)

        # 2. Migrate to GPU
        # IMPORTANT: .detach().requires_grad_(True) makes these new "Leaf Nodes" on the GPU
        x_gpu = x_cpu.to(device).detach().requires_grad_(True)
        t_gpu = t_cpu.to(device).detach().requires_grad_(True)

        # 3. Forward
        out = model(x_gpu, t_gpu)
        loss = out.mean()

        # 4. Backward
        loss.backward()

        # 5. Check Gradients on the GPU tensors
        self.assertIsNotNone(x_gpu.grad, "Gradient failed to populate on migrated input")
        self.assertGreater(x_gpu.grad.sum().item(), 1e-6, "Gradient is zero")

        # Check that weights still got gradients
        self.assertIsNotNone(model.linear1.weights.grad, "Weight gradients broke during migration")

if __name__ == '__main__':
    unittest.main()