import torch
import torch.nn as nn
import torch.nn.functional as F

class EDM_LOSS(nn.Module):
    def __init__(self):
        super().__init__()