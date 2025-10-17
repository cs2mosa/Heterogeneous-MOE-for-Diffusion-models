from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize(x: torch.Tensor, dim: Optional[list[int]] = None, eps: int = 1e-4)-> torch.Tensor:
    if dim is None:
        dim = list(range(1, x.ndim))
    w = torch.linalg.vector_norm(x,dim = dim, keepdim = True, dtype = torch.float32)
    w = torch.add(eps,w, alpha= np.sqrt(w.numel() / x.numel()))
    return x / w.to(x.dtype)

#magnitude preserving silu as in EDM2 paper
def mp_silu(x: torch.Tensor)-> torch.Tensor:
    return F.silu(x) / 0.596

#magnitude preserving sum as in EDM2 paper
def mp_sum(a: torch.Tensor,b: torch.Tensor,t: float)-> torch.Tensor:
    return a.lerp(b,t) / np.sqrt((1 - t)**2 + t**2)

#magnitude preserving concatenation as in EDM2 paper
def mp_cat(a: torch.Tensor,b:torch.Tensor,dim,t: float)-> torch.Tensor :
    Na = a.shape[dim]
    Nb = b.shape[dim]
    c1 = np.sqrt((Na+Nb)/((1-t)**2 + t**2))
    Wa = (c1* (1-t)) / np.sqrt(Na)
    Wb = (c1* t) / np.sqrt(Nb)
    return torch.cat([Wa*a, Wb*b],dim = dim)

#magnitude preserving fourier features as in EDM2 paper
class MP_Fourier(nn.Module):
    def __init__(self,num_channels: int, bandwidth: int = 1 ):
        super().__init__()
        self.register_buffer('freqs', 2 * torch.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * torch.pi * torch.rand(num_channels))

    def forward(self,x: torch.Tensor)-> torch.Tensor:
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)

#magnitude preserving convolution layer as in EDM2 paper
class MP_Conv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels,in_channels,*kernel))
        assert self.weights.numel() != 0
        self.kernel = kernel
    def forward(self,x: torch.Tensor, gain: int= 1)-> torch.Tensor:
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
    def __init__(self,
                 num_heads: int ,
                 emb_dim: int,
                 context_dim :Optional[int] = None,
                 attn_balance:int = 0.5,
                 res_balance: int = 0.5
                 ):
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

    def forward(self,
                query: torch.Tensor,
                gain: float,
                context: Optional[torch.Tensor] = None
                )-> torch.Tensor:

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