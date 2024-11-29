import wandb
import torch
import numpy as np
from tqdm import tqdm

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

def build_run_name(hyperparameters: dict, model_name: str):
    """Build a run name from a dictionary of hyperparameters."""
    return f"w_{hyperparameters['WHISPER_MODEL']}-dataset_{hyperparameters['DATASET_NAME']}-model_{model_name}-projdimfactor_{hyperparameters['PROJECTION_DIM_FACTOR']}-{'withcoords' if hyperparameters['COORDS'] else 'withoutcoords'}"
