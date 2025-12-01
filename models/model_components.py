from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model_internals as m

class Scaling_router(nn.Module):
    """
    A Soft-Gating Network (Router) that generates continuous scaling factors for experts.

    Unlike a Top-K router, this module assigns a weight to every expert. It is designed
    to analyze the input image features and determine how much 'gain'
    each expert path should receive.

    Architecture:
        Input -> CNN (Conv-BN-ReLU x2) -> Global Avg Pool -> Dropout -> Linear -> Softmax * 2
    """
    def __init__(self,
                 in_channels: Optional[int] = 3,
                 num_experts: Optional[int] = 2,
                 dropout :Optional[float] = 0.2
                 ):
        """
        Args:
            in_channels (int): Channels of the input feature map/image.
            num_experts (int): Number of scaling factors to output (one per expert).
            dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.soft_route = nn.Sequential(
            m.MP_Conv(in_channels=in_channels, out_channels=in_channels * 2, kernel=(3, 3)),# padding = same in all MP_conv
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            m.MP_Conv(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel=(3, 3)),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout(dropout)
        )
        self.linear = m.MP_Conv(in_channels=in_channels * 4, out_channels= num_experts, kernel=())

    def forward(self, x: torch.Tensor, zeta: Optional[float] = 1e-2) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor .(typically a noise conditioning vector shaped as a 4d tensor)
            zeta (float): Noise magnitude. Used to encourage exploration during training.
                          Should ideally decay over time (e.g., 1e-2 -> 0).
        Note: zeta should be inversely proportional with the number of training steps, using exponential decay for zeta in the training loop

        Returns:
            torch.Tensor: A tensor of shape (Batch, Num_Experts) containing scaling factors.
        """
        batch_size, in_channels, height, width = x.size()
        # output probability dist. over the experts to use as weights multiplied with the input and passed through the next stage
        x = self.soft_route(x)
        x = x.view(batch_size, -1)
        # passing through a linear layer for projection on the selection space
        x = self.linear(x)
        if self.training:
            x += torch.randn_like(x) * zeta
        scaling_output = F.softmax(x,dim = -1) * 2
        # making sure that the dominant path gets scaled up while the other gets scaled down
        return scaling_output

class Router(nn.Module):
    """
    A Sparse Top-K Gating Network (Router).

    This module selects a subset (Top-K) of experts to process the input.
    It returns a sparse weight vector (zeros for unselected experts) and the
    raw probabilities (useful for load-balancing auxiliary losses).

    Architecture:
        Input -> Deeper CNN (Conv-BN-ReLU x3) -> Global Avg Pool -> Linear -> TopK Selection
    """
    def __init__(self,
                 in_channels: Optional[int] = 3,
                 top_k: Optional[int] = 1,
                 num_experts: Optional[int] = 5,
                 dropout :Optional[float] = 0.2
                 ):
        """
        Args:
            in_channels (int): Input channels.
            top_k (int): How many experts to select active per sample.
            num_experts (int): Total number of available experts.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.hard_route = nn.Sequential(
            m.MP_Conv(in_channels = in_channels, out_channels= in_channels * 2,kernel=(3,3)), #padding = same in all MP_conv
            nn.BatchNorm2d(in_channels * 2),
            nn.ReLU(),
            m.MP_Conv(in_channels=in_channels * 2, out_channels=in_channels * 4, kernel=(3, 3)),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(),
            m.MP_Conv(in_channels = in_channels * 4, out_channels= in_channels * 4,kernel=(3,3)),
            nn.BatchNorm2d(in_channels * 4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(dropout)
        )
        self.linear = m.MP_Conv(in_channels=in_channels * 4, out_channels= num_experts, kernel=())
        self.k = top_k

    def forward(self,x:torch.Tensor, mask: Optional[torch.Tensor] | None, zeta: Optional[float] = 1e-2)->tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor. (typically a noise conditioning vector shaped as a 4d tensor)
            zeta (float): Noise magnitude for exploration.
            mask (torch.Tensor): mask for enhancing expert specialization, typically has dimensions as (batch , num_experts).
        Note:
            zeta should be inversely proportional with the number of training steps, using exponential decay for zeta in the training loop
        Returns:
            tuple:
                - sparse_gate_weights (torch.Tensor): Shape (B, Num_Experts).
                  Contains weights for Top-K experts, 0.0 for others.
                - gate_probs (torch.Tensor): Shape (B, Num_Experts).
                  Full probability distribution (Softmax) across all experts.
                  Used for calculating auxiliary load-balancing loss.
        """
        batch_size, in_channels, height, width = x.size()
        x = self.hard_route(x)
        #shape before (batch_size, in_channels_in, height, width) -> (batch_size, in_channels_out * height * width)
        x = x.view(batch_size,-1)
        #passing through a linear layer for projection on the selection space
        x = self.linear(x)
        #adding noise to encourage exploration in the early stages of training
        if self.training:
            x += torch.randn_like(x) * zeta

        #mask output logit for expert specialization
        x = x.masked_fill(mask == 0, float('-inf'))
        # for calculating the auxiliary loss
        gate_probs = F.softmax(x,dim = -1)
        topk_vals, topk_indices = torch.topk(x, self.k, dim=-1)
        gating_weights = F.softmax(topk_vals,dim = -1)
        #sparse representation of the gate weights-> placing the new gate weights in their place of the same dimensions as the linear layer output
        sparse_gate_weights = torch.zeros_like(x).scatter(-1, topk_indices, gating_weights)
        return sparse_gate_weights, gate_probs


class Unet_block(nn.Module):
    """
    the building block of the Unet experts in the Unet pathway , according to Nvidia's paper
        Note: we made the kernel size a variable input to enforce architectural differences between each expert in the Unet path
    """
    def __init__(self,
                 in_channels: int ,
                 out_channels: int ,
                 kernel: tuple,
                 emb_size: int ,
                 resample: Optional[str] = 'keep',
                 Type: Optional[str] = 'enc',
                 residual_balance: Optional[float] = 0.5,
                 Dropout: Optional[float] = 0.2
                 ):
        """
        Args:
            :param in_channels: number of input channels
            :param out_channels: number of desired output channels
            :param kernel: kernel size -> this is variable for every expert ... the only difference between this block and Nvidia's block
            :param emb_size: embedding dimensions
            :param resample:
                    keep: returns the same input (identity function)
                    up: upsamples the input by (2 x 2)
                    down: downsamples the input by (2 x 2)
            :param Type: encoder or decoder -> enc | dec
                Note: choosing 'enc' implies choosing 'down' in type ... to be consistent with Nvidia's diagram
            :param residual_balance: balance between the main branch and the residual branch of the block
            :param Dropout: usually 0.2
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_size = emb_size
        self.residual_balance= residual_balance
        self.type = Type
        self.resample = resample
        self.kernel = kernel
        self.dropout =  Dropout
        self.emb_gain = nn.Parameter(torch.zeros([]))
        self.conv_gain1 = nn.Parameter(torch.zeros([]))
        self.conv_gain2 = nn.Parameter(torch.zeros([]))
        self.conv_skip = m.MP_Conv(in_channels = in_channels, out_channels=out_channels,kernel=(1, 1)) if in_channels != out_channels else None
        self.emb_layer = m.MP_Conv(in_channels = emb_size, out_channels = out_channels,kernel=())
        self.conv_res1 = m.MP_Conv(in_channels = out_channels if self.type == 'enc' else in_channels, out_channels= out_channels, kernel=self.kernel)
        self.conv_res2 = m.MP_Conv(in_channels = out_channels, out_channels= out_channels, kernel=self.kernel)

    def forward(self, x:torch.Tensor, embedding:torch.Tensor) -> torch.Tensor:
        """
        the forward method of the Unet block
        Args:
            :param x: the input image tensor of dimensions -> (batch , in_channels, height, width)
            :param embedding: embedding tensor for text and noise merged before the entry , dimensions -> (batch, embedding_size)
                Note: concatenation with the skip connection must be done in the Full Unet class
            :return: processed output tensor of dimensions -> (batch , out_channels, new_height, new_width)
        """
        embedding = 1 + self.emb_layer(embedding,gain = self.emb_gain)
        #up for decoders : down for encoders
        x = m.resample(x, mode=self.resample)
        if self.type == 'enc':
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = m.normalize(x,dim = [1])

        main_branch = self.conv_res1(m.mp_silu(x),gain = self.conv_gain1)
        #embedding dimensions (batch,embedding_size) -> (batch, intermediate_channels,1,1) for broadcasting
        main_branch = main_branch * embedding.unsqueeze(2).unsqueeze(3).to(x.dtype)
        main_branch = m.mp_silu(main_branch)
        #dropout only in training mode
        if self.training and self.dropout != 0:
            main_branch = F.dropout(main_branch, p= self.dropout)

        main_branch = self.conv_res2(main_branch,gain = self.conv_gain2)
        #conv_skip for decoders in the residual branch
        if self.type == 'dec' and self.conv_skip is not None :
            x = self.conv_skip(x)

        return m.mp_sum(x,main_branch,t = self.residual_balance)


class Vit_block(nn.Module):
    def __init__(self,
                 num_heads:int ,
                 in_channels: int ,
                 out_channels: int ,
                 res_balance: float = 0.5 ,
                 ):
        super().__init__()

