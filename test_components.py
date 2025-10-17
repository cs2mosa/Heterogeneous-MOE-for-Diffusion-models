from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest


def normalize(x: torch.Tensor, dim: Optional[list[int]] = None, eps: int = 1e-4)-> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    w = torch.linalg.vector_norm(x,dim = dim, keepdim = True, dtype = torch.float32)
    w = torch.add(eps,w, alpha= np.sqrt(w.numel() / x.numel()))
    return x / w.to(x.dtype)

#magnitude preserving silu as in EDM2 paper
def mp_silu(x)-> torch.Tensor:
    return F.silu(x) / 0.596

#magnitude preserving sum as in EDM2 paper
def mp_sum(a,b,t)-> torch.Tensor:
    return a.lerp(b,t) / np.sqrt((1 - t)**2 + t**2)

#magnitude preserving concatenation as in EDM2 paper
def mp_cat(a,b,dim,t)-> torch.Tensor :
    Na = a.shape[dim]
    Nb = b.shape[dim]
    c1 = np.sqrt((Na+Nb)/((1-t)**2 + t**2))
    Wa = (c1* (1-t)) / np.sqrt(Na)
    Wb = (c1* t) / np.sqrt(Nb)
    return torch.cat([Wa*a, Wb*b],dim = dim)

#magnitude preserving fourier features as in EDM2 paper
class MP_Fourier(nn.Module):
    def __init__(self,num_channels: int, bandwidth: int= 1 ):
        super().__init__()
        self.register_buffer('freqs', 2 * torch.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * torch.pi * torch.rand(num_channels))

    def forward(self,x)-> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#magnitude preserving convolution layer as in EDM2 paper
class MP_Conv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple):
        super().__init__()
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels,in_channels,*kernel))
        assert self.weights.numel() != 0
        self.kernel = kernel
    def forward(self,x: torch.Tensor,gain: int= 1)-> torch.Tensor:
        w = self.weights.to(torch.float32)
        if self.training:
            with torch.no_grad():
                w = self.weights.copy_(normalize(w))
        w = (normalize(w) * (gain / np.sqrt(self.weights[0].numel()))).to(x.dtype)
        if x.ndim == 2 :
            return x @ w.t()
        assert x.ndim == 4
        return F.conv2d(x,w,padding = (w.shape[-1]//2,))

#magnitude preserving attention (single and multi_head) layer as in EDM2 paper
class MP_Attention(nn.Module):
    def __init__(self,num_heads: int ,emb_dim: int, context_dim :Optional[int] = None, attn_balance:int = 0.5,res_balance: int = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads

        assert emb_dim % num_heads == 0
        if context_dim is None:
            context_dim = emb_dim

        self.attn_balance = attn_balance
        self.res_balance = res_balance
        self.q_proj = MP_Conv(emb_dim,emb_dim,(1,1))
        self.k_proj = MP_Conv(context_dim,emb_dim,(1,1))
        self.v_proj = MP_Conv(context_dim,emb_dim,(1,1))
        self.out_proj = MP_Conv(emb_dim,emb_dim,(1,1))

    def forward(self,query: torch.Tensor, gain: float,context: Optional[torch.Tensor] = None)-> torch.Tensor:
        res = query
        batch_size,seq_len,emb_dim =query.shape
        context = query if context is None else context

        #emb_size = channels > because we need a different filter for each emb_size vector (the image)
        #out_size = batch_size,emb_dim,seq_len,1
        query = query.permute(0, 2, 1).unsqueeze(-1)
        context = context.permute(0, 2, 1).unsqueeze(-1)

        q_proj = self.q_proj(query,gain = gain)
        k_proj = self.k_proj(context,gain = gain)
        v_proj = self.v_proj(context,gain = gain)

        #shape matches (batch,num_heads,seq_ln,head_dim)
        q_proj = q_proj.view(batch_size,self.num_heads,self.head_dim,-1).transpose(-1,-2)
        k_proj = k_proj.view(batch_size,self.num_heads,self.head_dim,-1).transpose(-1,-2)
        v_proj = v_proj.view(batch_size,self.num_heads,self.head_dim,-1).transpose(-1,-2)

        #normalization across head_dim (originally emb_dim)
        q_proj = normalize(q_proj,dim = [3])
        k_proj = normalize(k_proj,dim = [3])

        #attention matrix multiplication
        q_k_prod = torch.einsum('bnsh,bnsk->bnhk',q_proj,k_proj / np.sqrt(q_proj.shape[2])).softmax(dim = 3)
        q_k_v_prod = torch.einsum('bnhk,bnsk->bnsh',q_k_prod,v_proj)

        attn_output = q_k_v_prod.transpose(1, 2).contiguous().view(batch_size, seq_len, emb_dim)
        attn_output_reshaped = attn_output.permute(0, 2, 1).unsqueeze(-1)

        out = self.out_proj(attn_output_reshaped, gain=gain)
        out = out.squeeze(-1).permute(0, 2, 1)
        return mp_sum(res,out,self.attn_balance)




# --- Unit Test Suite ---

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
        # Use a generous tolerance for stochastic variance checks
        self.tolerance = 1e-1

    def test_mp_silu(self):
        """Test magnitude-preserving SiLU."""
        x = torch.randn(self.batch_size, self.channels, self.height, self.width)
        self.assertAlmostEqual(np.sqrt(x.var().item()), 1.0, delta=self.tolerance)

        out = mp_silu(x)

        self.assertEqual(x.shape, out.shape)
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_sum(self):
        """Test magnitude-preserving sum (lerp)."""
        a = torch.randn(self.batch_size, self.channels)
        b = torch.randn(self.batch_size, self.channels)
        t = 0.75  # Blending factor

        out = mp_sum(a, b, t)

        self.assertEqual(a.shape, out.shape)
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_cat(self):
        """Test magnitude-preserving concatenation."""
        a = torch.randn(self.batch_size, self.channels, self.height, self.width)
        b = torch.randn(self.batch_size, self.channels // 2, self.height, self.width)
        t = 0.5
        dim = 1  # Concatenate on the channel dimension

        out = mp_cat(a, b, dim=dim, t=t)

        expected_channels = a.shape[dim] + b.shape[dim]
        self.assertEqual(out.shape, (self.batch_size, expected_channels, self.height, self.width))
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_fourier(self):
        """Test magnitude-preserving Fourier features."""
        num_features = 128
        fourier_layer = MP_Fourier(num_channels=num_features)
        x = torch.randn(self.batch_size)  # Input is typically a 1D tensor of timesteps

        out = fourier_layer(x)

        self.assertEqual(out.shape, (self.batch_size, num_features))
        self.assertAlmostEqual(np.sqrt(out.var().item()), 1.0, delta=self.tolerance)

    def test_mp_conv_4d(self):
        """Test magnitude-preserving 2D convolution (4D tensor)."""
        in_channels = self.channels
        out_channels = self.channels * 2
        kernel_size = (3,3)
        conv_layer = MP_Conv(in_channels, out_channels, kernel=kernel_size)
        lin_layer = MP_Conv(in_channels, out_channels, kernel=[])
        conv_layer.train()  # Ensure weights are copied
        lin_layer.train()
        x = torch.randn(self.batch_size, in_channels, self.height, self.width)
        y = torch.randn(self.batch_size, in_channels)
        out = conv_layer(x)
        out2 = lin_layer(y)
        self.assertEqual(out2.shape ,(self.batch_size,out_channels))
        self.assertEqual(out.shape,(self.batch_size, out_channels, self.height, self.width))
        self.assertAlmostEqual(np.sqrt(out.var().item()), np.sqrt(x.var().item()), delta=self.tolerance)

    def test_mp_attention(self):
        """Test magnitude-preserving Multi-Head Attention."""
        num_heads = 8
        attn_layer = MP_Attention(num_heads=num_heads, emb_dim=self.emb_dim,context_dim=self.emb_dim)

        # Test self-attention where query, key, and value are the same
        x = torch.randn(self.batch_size, self.seq_len, self.emb_dim)
        context = torch.randn(self.batch_size, self.seq_len,self.emb_dim)
        out = attn_layer(x,context = context, gain = 1)

        self.assertEqual(out.shape, x.shape)

        self.assertAlmostEqual(np.sqrt(out.var().item()),np.sqrt(0.5), delta=self.tolerance)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)