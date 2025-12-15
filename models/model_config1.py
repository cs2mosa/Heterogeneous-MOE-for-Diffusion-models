import torch
import torch.nn as nn
from models import model_internals as util
from models import model_components as m
import torch.nn.functional as F
from typing import Optional ,List,Tuple

#helper to abstract the router running operation
def router_to_unet_experts(x: torch.Tensor,
                      experts: nn.ModuleList,
                      out_router: torch.Tensor,
                      time_emb: torch.Tensor,
                      text_emb: torch.Tensor,
                      ) -> torch.Tensor:

    if text_emb is not None and text_emb.ndim == 3:
        # Average over the sequence dimension (dim 1)
        text_processed = text_emb.mean(dim=1)
    else:
        text_processed = text_emb
    output = torch.zeros_like(x)

    for i, expert in enumerate(experts):
        mask = out_router[:, i] > 0

        if not mask.any():
            continue

        x_expert = x[mask]
        time_expert = time_emb[mask]
        text_expert = text_processed[mask] if text_processed is not None else None
        out_exp = expert(x=x_expert, time_emb=time_expert, text_emb=text_expert)
        router_weights = out_router[mask, i]
        router_weights = router_weights.view(-1, 1, 1, 1)
        output[mask] += out_exp * router_weights

    return output

#adding gating MLP network after the attention module with 2 units output and softmax activation -> linear interpolation with the 2 units across the out_unet and Out_VIT
class HDMOEM (nn.Module):

    def __init__(self,
                 IN_in_channels       : int,  # in_channels == Out_channels in VIT experts and Unet experts
                 IN_img_resolution    : int,  # seq_ln  = img_res ** 2
                 time_emb_dim         : int,
                 text_emb_dim         : int,
                 num_experts          : int,
                 top_k                : int,
                 Fourier_bandwidth    : float,
                 VIT_num_blocks       : int,
                 VIT_patch_sizes      : List[int], #patches are square small pieces of image
                 VIT_num_groups       : int,
                 VIT_num_heads        : int,
                 VIT_emb_size         : int,  # internal VIT embedding size
                 Unet_num_blocks      : int,
                 Unet_channel_mult    : list[int], #shape of (num_experts , levels)
                 Unet_kernel_sizes    : List[Tuple[int, int]],
                 Unet_model_channels  : Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance   : Optional[float] = 0.5,
                 Unet_concat_balance  : Optional[float] = 0.5,
                 ):
        """

        :param IN_in_channels: input image channels (typically 4 for a VAE latent diffusion)
        :param IN_img_resolution: input image resolution (we deal with square images only in this implementation -> height = width)
        :param time_emb_dim: expected embedding dimensions of time vector
        :param text_emb_dim: expected embedding dimensions of time vector
        :param num_experts: number of expert per path
        :param top_k: router's topk -> 1 = switch router
        :param Fourier_bandwidth: default 1.0
        :param VIT_num_blocks: number of vision transformer blocks per VIT expert
        :param VIT_patch_sizes: list of patch sizes -> higher patch_size for high noise and vice versa
        :param VIT_num_groups: number of groups for group normalizations in VIT experts
        :param VIT_num_heads: number of attention heads per VIT expert
        :param VIT_emb_size: internal embedding dimension for each VIT expert
        :param Unet_num_blocks: number of Unet blocks per Unet expert
        :param Unet_channel_mult: list of channel multipliers for progressive increase in channels -> going in depth of each Unet expert
        :param Unet_kernel_sizes: list of kernel sizes (one kernel for each expert). typically going higher in kernel sizes while progressing the list
        :param Unet_model_channels: treated as embedding size of the VIT. typically 192 channels is enough
        :param Unet_channel_mult_emb: list of embedding channel multipliers for each unet expert
        :param Unet_label_balance: text balance of Unet
        :param Unet_concat_balance: concatenation balance of Unet
        """
        super().__init__()
        #assert IN_img_resolution ** 2 == VIT_seq_ln
        self.Fourier_emb = util.MP_Fourier(num_channels=time_emb_dim // 2,
                                           bandwidth=Fourier_bandwidth)

        self.out_fourier1 = util.MP_Conv(in_channels= time_emb_dim // 2,
                                         out_channels= time_emb_dim * 2,
                                         kernel=())

        self.out_fourier2 = util.MP_Conv(in_channels=time_emb_dim * 2,
                                         out_channels=time_emb_dim,
                                         kernel=())

        self.scaling_net = m.Scaling_router(emb_dim= time_emb_dim,
                                            num_experts= 2)

        self.Unet_router = m.Router(in_channels= IN_in_channels,
                                    time_dim= time_emb_dim,
                                    top_k= top_k,
                                    num_experts= num_experts)

        self.vit_router = m.Router(in_channels=IN_in_channels,
                                    time_dim=time_emb_dim,
                                    top_k=top_k,
                                    num_experts= num_experts)

        self.Unet_experts= nn.ModuleList()
        for i in range(num_experts):
            self.Unet_experts.append(m.Unet_expert(img_resolution= IN_img_resolution,
                                                   img_channels= IN_in_channels,
                                                   time_emb_dim= time_emb_dim,
                                                   text_emb_dim= text_emb_dim,
                                                   num_blocks=Unet_num_blocks,
                                                   channel_mult= Unet_channel_mult,
                                                   kernel_size=Unet_kernel_sizes[i],
                                                   label_balance=Unet_label_balance,
                                                   concat_balance=Unet_concat_balance,
                                                   model_channels=Unet_model_channels,
                                                   channel_mult_emb=Unet_channel_mult_emb
                                                   ))
        self.VIT_experts = nn.ModuleList()
        for i in range(num_experts):
            self.VIT_experts.append(m.Vit_expert(num_heads= VIT_num_heads,
                                                  num_groups=VIT_num_groups,
                                                  in_channels=IN_in_channels,
                                                  seq_ln= (IN_img_resolution // VIT_patch_sizes[i]) ** 2,
                                                  emb_dim=VIT_emb_size,
                                                  num_blocks= VIT_num_blocks,
                                                  patch_size= VIT_patch_sizes[i],
                                                  text_dim= text_emb_dim,
                                                  time_dim= time_emb_dim
                                                  ))

        self.cross_attn = util.MP_Attention(num_heads= VIT_num_heads,
                                                  emb_dim= IN_in_channels,
                                                  seq_ln= IN_img_resolution ** 2,
                                                  context_dim= IN_in_channels,
                                                  attn_balance= 0.8,
                                                  is_cross_attn= True)

        self.gate1 = util.MP_Conv(in_channels= IN_in_channels * 2,
                                  out_channels=IN_in_channels ,
                                  kernel=(1,1)
                                  )
        self.gate2 = util.MP_Conv(in_channels= IN_in_channels ,
                                  out_channels = 2,
                                  kernel=(1,1)
                                  )

    def forward(self,
                x               : torch.Tensor, #shape -> (batch,in_channels,height, width)
                time_vec        : torch.Tensor, #sigma in literature shape ->(batch,)
                text_emb        : torch.Tensor, #shape -> (batch,text_seq_ln,text_emb_dim)
                Unet_router_mask: torch.Tensor, #shape-> (batch,num_experts)
                Vit_router_mask : torch.Tensor, #shape-> (batch,num_experts)
                zeta            : float
                )-> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:

        B,C,H,W = x.shape
        time_embed = self.Fourier_emb(x = time_vec)
        time_embed = self.out_fourier1(x = time_embed)
        time_embed = self.out_fourier2(x = util.mp_silu(x= time_embed))
        scaling_factors = self.scaling_net(x = time_embed,zeta = zeta)
        scaling_vit = scaling_factors[:,0:1].view(-1, 1, 1, 1)
        scaling_Unet = scaling_factors[:,1:2].view(-1, 1, 1, 1)
        in_unet_router = scaling_Unet * x
        in_vit_router = scaling_vit * x
        out_vit_router,Vit_gate_probs = self.vit_router(x = in_vit_router,
                                         time_emb = time_embed,
                                         zeta= zeta,
                                         mask = Vit_router_mask)

        out_unet_router,Unet_gate_probs = self.Unet_router(x = in_unet_router,
                                           time_emb = time_embed,
                                           zeta = zeta,
                                           mask = Unet_router_mask)

        out_Unet_expert = router_to_unet_experts(x = in_unet_router,
                                            experts=self.Unet_experts,
                                            time_emb= time_embed,
                                            text_emb= text_emb,
                                            out_router=out_unet_router
                                            )

        out_Vit_expert = router_to_unet_experts(x = in_vit_router,
                                           experts=self.VIT_experts,
                                           time_emb= time_embed,
                                           text_emb= text_emb,
                                           out_router=out_vit_router
                                            )

        query = out_Unet_expert.flatten(2).transpose(1, 2)
        context = out_Vit_expert.flatten(2).transpose(1, 2)
        out_final_attn = self.cross_attn(query = query,
                                          context = context,
                                          gain_s = 1.0,
                                          gain_t = 1.0,
                                         )

        out_final_attn_img = out_final_attn.transpose(1, 2).view(B,C,H,W)
        in_gate = util.mp_cat(out_Unet_expert,out_final_attn_img,dim = 1)
        out_gate = self.gate1(in_gate)
        out_gate = self.gate2(util.mp_silu(out_gate))
        out_gate = F.softmax(out_gate,dim = 1)
        Wx = out_gate[:,0:1]
        Wa = out_gate[:,1:2]
        out_gated_attn = Wx * out_Unet_expert + Wa *  out_final_attn_img

        return out_gated_attn, Unet_gate_probs, Vit_gate_probs


class preconditioned_HDMOEM(nn.Module):
    def __init__(self,
                 IN_in_channels: int,  # in_channels == Out_channels in VIT experts and Unet experts
                 IN_img_resolution: int,  # seq_ln  = img_res ** 2
                 time_emb_dim: int,
                 text_emb_dim: int,
                 num_experts: int,
                 top_k: int,
                 Fourier_bandwidth: float,
                 VIT_num_blocks: int,
                 VIT_patch_sizes: List[int],  # patches are square small pieces of image
                 VIT_num_groups: int,
                 VIT_num_heads: int,
                 VIT_emb_size: int,  # internal VIT embedding size
                 Unet_num_blocks: int,
                 Unet_channel_mult: list[int],  # shape of (num_experts , levels)
                 Unet_kernel_sizes: List[Tuple[int, int]],
                 Unet_model_channels: Optional[int] = 192,
                 Unet_channel_mult_emb: Optional[int] = None,
                 Unet_label_balance: Optional[float] = 0.5,
                 Unet_concat_balance: Optional[float] = 0.5,
                 sigma_data: Optional[float] = 0.5, #expected standard deviation of the data
                 log_var_channels: Optional[int] = 128
                 ):
        super().__init__()
        self.sigma_data = sigma_data
        self.log_var_channels = log_var_channels
        self.log_var_fourier = util.MP_Fourier(num_channels= self.log_var_channels)
        self.log_var_linear = util.MP_Conv(in_channels= self.log_var_channels, out_channels=1, kernel=())
        self.net = HDMOEM(IN_in_channels = IN_in_channels,
                          IN_img_resolution = IN_img_resolution,  # seq_ln  = img_res ** 2
                          time_emb_dim= time_emb_dim,
                          text_emb_dim =text_emb_dim,
                          num_experts =num_experts,
                          top_k = top_k,
                          Fourier_bandwidth =Fourier_bandwidth,
                          VIT_num_blocks = VIT_num_blocks,
                          VIT_patch_sizes=VIT_patch_sizes,  # patches are square small pieces of image
                          VIT_num_groups=VIT_num_groups,
                          VIT_num_heads=VIT_num_heads,
                          VIT_emb_size =VIT_emb_size,  # internal VIT embedding size
                          Unet_num_blocks= Unet_num_blocks,
                          Unet_channel_mult=Unet_channel_mult,  # shape of (num_experts , levels)
                          Unet_kernel_sizes =Unet_kernel_sizes,
                          Unet_model_channels =Unet_model_channels,
                          Unet_channel_mult_emb = Unet_channel_mult_emb,
                          Unet_label_balance= Unet_label_balance,
                          Unet_concat_balance = Unet_concat_balance
                          )

    def forward(self,
                x: torch.Tensor,  # shape -> (batch,in_channels,height, width)
                sigma: torch.Tensor,  # sigma in literature shape ->(batch,)
                text_emb: torch.Tensor,  # shape -> (batch,text_seq_ln,text_emb_dim)
                Unet_router_mask: torch.Tensor,  # shape-> (batch,num_experts)
                Vit_router_mask: torch.Tensor,  # shape-> (batch,num_experts)
                zeta: float,
                return_log_var:bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor] | None]:

        # Preconditioning weights.
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out =  sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.flatten().log() / 4
        x = x * c_in
        out_net,Unet_gate_probs, Vit_gate_probs = self.net(x = x,
                           text_emb = text_emb,
                           time_vec = c_noise ,
                           Unet_router_mask = Unet_router_mask,
                           Vit_router_mask = Vit_router_mask,
                           zeta = zeta)
        D_x = c_skip * x + c_out* out_net
        if return_log_var:
            log_var = self.log_var_linear(self.log_var_fourier(c_noise)).reshape(-1, 1, 1, 1)
            return D_x, Unet_gate_probs, Vit_gate_probs, log_var
        else:
            return D_x, Unet_gate_probs, Vit_gate_probs, None