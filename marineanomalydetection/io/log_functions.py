import loguru
import numpy as np
import torch


def log_patches(
    patches: np.ndarray | list[str], 
    logger: loguru._logger.Logger
) -> None:
    """Logs the name of the patches

    Args:
        patches (np.ndarray | list[str]): names of the patches.
        logger (loguru._logger.Logger): logger.
    """
    for roi_print in patches:
        logger.info(roi_print)
    logger.info(f"Total of {len(patches)} training patches.")


def log_epoch_init(
    epoch: int, 
    logger: loguru._logger.Logger,
    separator: str = "_", 
    num_sep: int = 40
) -> None:
    """Logs the epoch number.

    Args:
        epoch (int): number of epoch.
        logger (loguru._logger.Logger): logger.
        separator (str, optional): separator. Defaults to "_".
        num_sep (int, optional): number of separators. Defaults to 40.
    """
    logger.info(
        separator * num_sep + "Epoch " + str(epoch) + ": " + separator * num_sep
    )


def log_ssl_loss_components(
    supervised_loss: float,
    unsupervised_loss: float,
    max_probs: torch.Tensor,
    mask: torch.Tensor,
    logger: loguru._logger.Logger,
    separator: str = "-",
    num_sep: int = 20
) -> None:
    """Logs:
      - the loss components (supervised and unsupervised) of the 
        semi-supervised loss.
      - the maximum softmax value among all predicted labels of the pixels of
        all patches in the last batch.

    Args:
        supervised_loss (float): supervised loss value.
        unsupervised_loss (float): unsupervised loss value.
        max_probs (torch.Tensor): maximum softmax value for all predicted 
          labels of each pixel of each patch in the batch.
        mask (torch.Tensor): mask with pixels to consider in the unsupervised 
          loss.
        logger (loguru._logger.Logger): logger.
        separator (str, optional): separator. Defaults to "-".
        num_sep (int, optional): num of separators. Defaults to 20.
    """
    print(max_probs[:, 0, 0])
    logger.info(separator * num_sep)
    logger.info(f"Lx: {supervised_loss}")
    logger.info(f"Lu: {unsupervised_loss}")
    logger.info(f"Max prob: {max_probs.max()}")
    logger.info(f"# pixels with confidence > threshold: {len(mask[mask == 1])}")
