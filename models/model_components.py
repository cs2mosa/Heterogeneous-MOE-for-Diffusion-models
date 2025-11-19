from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model_internals as m

class Scaling_router(nn.Module):
    def __init__(self,
                 in_channels: Optional[int] = 3,
                 num_experts: Optional[int] = 2,
                 dropout :Optional[float] = 0.2):

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
    def __init__(self,
                 in_channels: Optional[int] = 3,
                 top_k: Optional[int] = 1,
                 num_experts: Optional[int] = 5,
                 dropout :Optional[float] = 0.2):

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

    #zeta should be inversely proportional with the number of training steps
    def forward(self,x:torch.Tensor, zeta: Optional[float] = 1e-2)->tuple[torch.Tensor, torch.Tensor]:
        batch_size, in_channels, height, width = x.size()
        x = self.hard_route(x)
        #shape before (batch_size, in_channels_in, height, width) -> (batch_size, in_channels_out * height * width)
        x = x.view(batch_size,-1)
        #passing through a linear layer for projection on the selection space
        x = self.linear(x)
        #adding noise to encourage exploration in the early stages of training
        if self.training:
            x += torch.randn_like(x) * zeta
        #for calculating the auxiliary loss
        gate_probs = F.softmax(x,dim = -1)
        topk_vals, topk_indices = torch.topk(x, self.k, dim=-1)
        gating_weights = F.softmax(topk_vals,dim = -1)
        #sparse representation of the gate weights-> placing the new gate weights in their place of the same dimensions as the linear layer output
        sparse_gate_weights = torch.zeros_like(x).scatter(-1, topk_indices, gating_weights)
        return sparse_gate_weights, gate_probs


class Unet_block(nn.Module):
    def __init__(self,
                 in_channels: int ,
                 out_channels: int ,
                 emb_size: int ,
                 attention: Optional[bool] = False,
                 Type: Optional[str] = 'enc',
                 BottleNeck: Optional[bool] = False
                 ):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_size = emb_size
        self.attention = attention

class Vit(nn.Module):
    def __init__(self,
                 num_heads:int = 8,
                 input_channels: int = 3):
        super().__init__()
