import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Sets the seed to limit the number of sources of nondeterministic 
    behavior for reproducibility purposes.

    Args:
        seed (int): seed.
    """
    # Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_seed_worker(worker_id) -> None:
    """Sets the seed of DataLoader to limit the number of sources of 
    nondeterministic behavior for reproducibility purposes.
    """
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)