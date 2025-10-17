import torch
import torch.nn as nn
import torch.functional as F
import model_internals as m

class Router(nn.Module):
    def __init__(self):
        super().__init__()

class Unet_block(nn.Module):
    def __init__(self,
                 in_channels: int ,
                 out_channels: int ,
                 emb_size: int ,
                 attention: bool = False,
                 Type: str = 'enc'
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

