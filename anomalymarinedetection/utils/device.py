import torch

def get_device() -> torch.device:
    """Initializes the device.

    Returns:
        torch.device: device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device

def empty_cache() -> None:
    """Releases all unoccupied cached memory currently held by the caching 
    allocator so that those can be used in other GPU application and visible
    in nvidia-smi.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()