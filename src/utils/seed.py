import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set deterministic seed across all random number generators.

    Args:
        seed: Integer seed value. Default is 42.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        # Documented PyTorch behavior: some CUDA/CPU kernels lack deterministic implementations
        pass


def seed_worker(worker_id: int) -> None:
    """
    DataLoader worker initialization function for deterministic batching.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
