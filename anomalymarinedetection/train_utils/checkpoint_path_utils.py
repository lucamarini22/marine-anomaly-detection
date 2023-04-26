import os

from anomalymarinedetection.trainmode import TrainMode


def update_checkpoint_path(mode: TrainMode, checkpoint_path: str) -> str:
    """_summary_

    Args:
        mode (TrainMode): training mode.
        checkpoint_path (str): path of the checkpoint.

    Raises:
        Exception: if the mode is not known. 

    Returns:
        str: the updated path of the checkpoint.
    """
    if mode == TrainMode.TRAIN_SSL:
        checkpoint_path = os.path.join(checkpoint_path, "semi-supervised")
    elif mode == TrainMode.TRAIN:
        checkpoint_path = os.path.join(checkpoint_path, "supervised")
    else:
        raise Exception("Mode is not known.")
    return checkpoint_path
    
def check_checkpoint_path_exist(checkpoint_path: str):
    """Checks that the checkpoint path is an existing folder.

    Args:
        checkpoint_path (str): path of the checkpoint.

    Raises:
        Exception: if checkpoint_path is not a directory.
    """
    if not os.path.isdir(checkpoint_path):
        raise Exception(
            f'The checkpoint directory {checkpoint_path} does not exist'
        )
    