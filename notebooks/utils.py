import torch
import numpy as np

def get_device():
    """Get the appropriate device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For M1/M2 Macs
    else:
        return torch.device("cpu")

def set_all_seeds(seed: int):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)