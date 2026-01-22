from typing import Any
from models.model_config1 import preconditioned_HDMOEM
import torch
from Utilities import EDM_LOSS, sample_sigma
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image

def training_HDMOE(train_configs: dict[str,Any]):
    pass
