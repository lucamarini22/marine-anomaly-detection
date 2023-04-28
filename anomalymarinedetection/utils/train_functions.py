import os
import ast
import torch
from torch import nn

from anomalymarinedetection.trainmode import TrainMode
from anomalymarinedetection.utils.assets import labels_binary, labels_multi
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)


def get_optimizer(
    model: nn.Module, lr: float, weight_decay: float
) -> torch.optim:
    """Gets the optimizer.

    Args:
        model (nn.Module): model.
        lr (float): learning rate.
        weight_decay (float): weight decay.

    Returns:
        torch.optim: optimizer.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return optimizer

def get_lr_scheduler(reduce_lr_on_plateau: int, optimizer: torch.optim, lr_steps: list[int]) -> torch.optim.lr_scheduler:
    """Gets the learning rate scheduler.

    Args:
        reduce_lr_on_plateau (int): reduces learning rate when no increase 
          (0 or 1).
        optimizer (torch.optim): optimizer.
        lr_steps (list[int]): the steps that the lr will be reduced.

    Returns:
        torch.optim.lr_scheduler: learning rate scheduler.
    """
    if reduce_lr_on_plateau == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, lr_steps, gamma=0.1, verbose=True
        )
    return scheduler

def check_num_alphas(alphas: torch.Tensor, output_channels: int) -> None:
    """Checks that the number of alpha coefficients is equal to the number of
    output channels.

    Args:
        alphas (torch.Tensor): alpha coefficients.
        output_channels (int): output channels.

    Raises:
        Exception: exception raised if the number of alpha coefficients is not
        equal to the number of output channels.
    """
    if len(alphas) != output_channels:
        raise Exception(
            f"There should be as many alphas as the number of categories, which in this case is {output_channels} because the parameter aggregate_classes was set to {options['aggregate_classes']}"
        )


def get_lr_steps(lr_steps: str) -> list:
    """Gets the learning rate steps at which decaying it.

    Args:
        lr_steps (str): learning ate steps.

    Raises:
        Exception: if the lr_steps are not specified as a string of a number
          or as a string of list of numbers

    Returns:
        list: the steps at which decaying the learning rate.
    """
    lr_steps = ast.literal_eval(lr_steps)
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise Exception('Please specify the lr_steps as "num" or as "[num]"')
    return lr_steps


def get_output_channels(aggregate_classes: CategoryAggregation) -> int:
    """_summary_

    Args:
        aggregate_classes (CategoryAggregation): _description_

    Raises:
        Exception: _description_

    Returns:
        int: _description_
    """
    if aggregate_classes == CategoryAggregation.MULTI:
        output_channels = len(labels_multi)
    elif aggregate_classes == CategoryAggregation.BINARY:
        output_channels = len(labels_binary)
    else:
        raise Exception(
            "The aggregated_classes option should be binary or multi"
        )
    return output_channels


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
            f"The checkpoint directory {checkpoint_path} does not exist"
        )
