from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def normalize(x: torch.Tensor, dim: Optional[list[int]] = None, eps: int = 1e-4)-> torch.Tensor:
    """
        Normalizes the input tensor `x` to have unit variance (RMS Normalization).

        This technique is used to force the input to lie on a hypersphere, ensuring
        consistent signal magnitude regardless of the specific values.

        Args:
            x (torch.Tensor): Input tensor of any shape (usually [N, C, ...]).
            dim (list[int], optional): Dimensions to reduce over. If None, reduces
                                       over all dimensions except the first (batch).
                                       Defaults to None.
            eps (float, optional): A small constant for numerical stability to avoid
                                   division by zero. Defaults to 1e-4.

        Returns:
            torch.Tensor: The normalized tensor with the same shape as `x`.
        """
    if dim is None:
        dim = list(range(1, x.ndim))
    w = torch.linalg.vector_norm(x,dim = dim, keepdim = True, dtype = torch.float32)
    w = torch.add(eps,w, alpha= np.sqrt(w.numel() / x.numel()))
    return x / w.to(x.dtype)

#magnitude preserving silu as in EDM2 paper
def mp_silu(x: torch.Tensor)-> torch.Tensor:
    """
        Applies the SiLU (Swish) activation function, scaled to preserve variance.

        Standard SiLU reduces the standard deviation of the signal. Dividing by
        0.596 restores the variance to 1 (assuming the input variance was 1).
        This is specific to the EDM2 (Elucidating the Design Space of Diffusion-Based Generative Models) architecture.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated tensor with preserved magnitude.
        """
    return F.silu(x) / 0.596

#magnitude preserving sum as in EDM2 paper
def mp_sum(a: torch.Tensor,b: torch.Tensor,t: float = 0.5)-> torch.Tensor:
    """
        Computes a weighted sum of tensors `a` and `b` that preserves variance.

        It performs linear interpolation followed by a normalization factor.
        Formula: result = ((1-t)*a + t*b) / sqrt((1-t)^2 + t^2)

        Args:
            a (torch.Tensor): First input tensor.
            b (torch.Tensor): Second input tensor.
            t (float, optional): Interpolation weight. 0.0 is all `a`, 1.0 is all `b`.
                                 Defaults to 0.5.

        Returns:
            torch.Tensor: The combined tensor with unit variance (assuming inputs were unit variance).
        """
    return a.lerp(b,t) / np.sqrt((1 - t)**2 + t**2)

#magnitude preserving concatenation as in EDM2 paper
def mp_cat(a: torch.Tensor,b:torch.Tensor,dim: int = 1,t: float = 0.5)-> torch.Tensor :
    """
        Concatenates two tensors `a` and `b` along a dimension while scaling them
        to preserve the overall magnitude/variance of the resulting feature vector.

        Instead of a simple concatenation where energy adds up, this weights `a`
        and `b` based on `t` so the resulting tensor maintains a stable magnitude statistic.

        Args:
            a (torch.Tensor): First input tensor.
            b (torch.Tensor): Second input tensor.
            dim (int, optional): Dimension along which to concatenate. Defaults to 1 (Channel dim).
            t (float, optional): Weight factor. A higher `t` gives more variance weight
                                 to `b` and less to `a`. Defaults to 0.5.

        Returns:
            torch.Tensor: The concatenated and scaled tensor.
        """
    Na = a.shape[dim]
    Nb = b.shape[dim]
    c1 = np.sqrt((Na+Nb)/((1-t)**2 + t**2))
    Wa = (c1* (1-t)) / np.sqrt(Na)
    Wb = (c1* t) / np.sqrt(Nb)
    return torch.cat([Wa*a, Wb*b],dim = dim)


def resample(x, f:Optional[list] = (1, 1), mode: Optional[str]='keep'):
    """
    Resamples a 4D tensor (Batch, Channels, Height, Width) using a separable filter.

    Args:
        x: Input tensor.
        f: Filter kernel (default bilinear [1,1]). Must be 1D and even length.
        mode: 'keep', 'down', or 'up'.
            keep: returns the input as it is.
            up: uses conv_transpose to upsample the input
            down: uses conv_ to downsample the input
    """
    if mode == 'keep':
        return x

    if not isinstance(f, torch.Tensor):
        f = torch.tensor(f, dtype=torch.float32)

    assert f.ndim == 1 and f.shape[0] % 2 == 0
    pad = (f.shape[0] - 1) // 2
    f = f / f.sum()
    f = torch.outer(f, f)
    f = f.unsqueeze(0).unsqueeze(0)
    f = f.to(device=x.device, dtype=x.dtype)
    c = x.shape[1]
    kernel = f.repeat(c, 1, 1, 1)
    if mode == 'down':
        return F.conv2d(x, kernel, stride=2, groups=c, padding=pad)

    if mode == 'up':
        return F.conv_transpose2d(x, kernel * 4, stride=2, groups=c, padding=pad)

    raise ValueError(f"Invalid mode: {mode}")

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
    """
    Magnitude Preserving Convolution/Linear Layer.

    This layer forces the weights to have unit norm and dynamically scales them
    based on the number of input elements (fan-in). It ensures that the variance
    of the output signal remains consistent with the input signal, preventing
    signal explosion or vanishing gradients.

    It handles both 4D inputs (Convolution) and 2D inputs (Linear/Dense).
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: tuple
                 ):
        """
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel (tuple): Kernel size, e.g., (3,3) for conv or () for linear.
                            Note: Code logic implies square/odd kernels for correct padding.
        """
        super().__init__()
        self.out_channels = out_channels
        self.weights = nn.Parameter(torch.randn(out_channels,in_channels,*kernel))
        assert self.weights.numel() != 0
        self.kernel = kernel

    def forward(self,x: torch.Tensor, gain: int= 1)-> torch.Tensor:
        """
        Forward pass with dynamic weight normalization and scaling.

        Args:
            x (torch.Tensor): Input tensor.
                              Shape (N, C) for Linear mode, or (N, C, H, W) for Conv mode.
            gain (float, optional): Scaling factor for the output magnitude.
                                    Defaults to 1.0.

        Returns:
            torch.Tensor: The convolved or projected output.
        """
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
    """
    Magnitude Preserving Multi-Head Attention.

    This block implements attention with specific normalizations to maintain signal variance
    (Magnitude Preserving). It differs from standard "Dot-Product" attention in two ways:

    1. It uses MP_Conv for projections to preserve magnitude.
    2. It computes "Channel Attention" (contracting over sequence length) rather than
       "Spatial Attention" (contracting over channel dimensions). This creates a
       (Head_Dim x Head_Dim) attention map instead of (Seq_Len x Seq_Len), making it
       much more memory efficient for long sequences (e.g., high-res images).
    """
    def __init__(self,
                 num_heads: int ,
                 emb_dim: int,
                 context_dim :Optional[int] = None,
                 attn_balance:int = 0.5,
                 ):
        """
        Args:
            num_heads (int): Number of attention heads.
            emb_dim (int): Total dimensionality of the input embedding.
            context_dim (int, optional): Dimensionality of the context (for Cross-Attention).
                                         If None, defaults to emb_dim (Self-Attention).
            attn_balance (float): Weight `t` for the residual connection in `mp_sum`.
                                  0.0 = Keep input, 1.0 = Use only attention output.
        """
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.head_dim = emb_dim // num_heads

        assert emb_dim % num_heads == 0
        if context_dim is None:
            context_dim = emb_dim

        self.attn_balance = attn_balance
        self.q_proj = MP_Conv(emb_dim,emb_dim,(1,1))
        self.k_proj = MP_Conv(context_dim,emb_dim,(1,1))
        self.v_proj = MP_Conv(context_dim,emb_dim,(1,1))
        self.out_proj = MP_Conv(emb_dim,emb_dim,(1,1))

    def forward(self,
                query: torch.Tensor,
                gain: float,
                context: Optional[torch.Tensor] = None
                )-> torch.Tensor:
        """
        Args:
            query (torch.Tensor): Input sequence. Shape (Batch, Seq_Len, Emb_Dim).
            gain (float): Magnitude scaling factor passed to MP_Conv.
            context (torch.Tensor, optional): Context for cross-attention.
                                              Shape (Batch, Seq_Len_Ctx, Context_Dim).

        Returns:
            torch.Tensor: Output tensor of shape (Batch, Seq_Len, Emb_Dim).
        """
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