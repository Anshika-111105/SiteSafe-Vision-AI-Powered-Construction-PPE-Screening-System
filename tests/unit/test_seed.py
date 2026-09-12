import random
import numpy as np
import torch
from src.utils.seed import set_seed


def test_seed_reproducibility():
    set_seed(42)
    val_py_1 = random.random()
    val_np_1 = np.random.rand()
    val_th_1 = torch.rand(1).item()

    set_seed(42)
    val_py_2 = random.random()
    val_np_2 = np.random.rand()
    val_th_2 = torch.rand(1).item()

    assert val_py_1 == val_py_2
    assert val_np_1 == val_np_2
    assert val_th_1 == val_th_2
