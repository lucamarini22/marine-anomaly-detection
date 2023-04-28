import os
import ast
from typing import Iterator
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from anomalymarinedetection.dataset.augmentation.discreterandomrotation import (
    DiscreteRandomRotation,
)
from anomalymarinedetection.dataset.augmentation.randaugment import (
    RandAugmentMC,
)
from anomalymarinedetection.dataset.augmentation.strongaugmentation import (
    StrongAugmentation,
)
from anomalymarinedetection.trainmode import TrainMode
from anomalymarinedetection.utils.assets import labels_binary, labels_multi
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.loss.focal_loss import FocalLoss
from anomalymarinedetection.models.unet import UNet
from anomalymarinedetection.utils.constants import (
    IGNORE_INDEX,
    ANGLES_FIXED_ROTATION,
    PADDING_VAL,
)
from anomalymarinedetection.io.file_io import FileIO


def train_step_semi_supervised(
    file_io: FileIO,
    labeled_train_loader: DataLoader,
    unlabeled_train_loader: DataLoader,
    labeled_iter: Iterator,
    unlabeled_iter: Iterator,
    criterion: nn.Module,
    criterion_unsup: nn.Module,
    training_loss: list[float],
    model: nn.Module,
    model_name: str,
    optimizer: torch.optim,
    epoch: int,
    device: torch.device,
    batch_size: int,
    classes_channel_idx: int,
    threshold: float,
    lambda_v: float,
    padding_val: int = PADDING_VAL,
) -> tuple[torch.Tensor, list[float]]:
    """Trains the model for one semi-supervised epoch.

    Args:
        file_io (FileIO): file io.
        labeled_train_loader (DataLoader): dataloader for labeled training set.
        unlabeled_train_loader (DataLoader): dataloader for unlabeled training
          set.
        labeled_iter (Iterator): iterator of labeled_train_loader.
        unlabeled_iter (Iterator): iterator of unlabeled_train_loader.
        criterion (nn.Module): supervised loss.
        criterion_unsup (nn.Module): unsupervised loss.
        training_loss (list[float]): list of semi-supervised training loss of
          batches.
        model (nn.Module): model.
        model_name (str): name of the model.
        optimizer (torch.optim): optimizer
        epoch (int): epoch.
        device (torch.device): device.
        batch_size (int): batch size.
        classes_channel_idx (int): index of the channel of the categories.
        threshold (float): threshold for model+s confidence (threshold for
          softmax values).
        lambda_v (float): coefficient of the unsupervised loss.
        padding_val (int, optional): padding value. Defaults to PADDING_VAL.

    Returns:
        tuple[torch.Tensor, list[float]]: last semi-superivsed loss, list of all
          semi-supervised losses of the current epoch.
    """
    try:
        # Load labeled batch
        img_x, seg_map = next(labeled_iter)
    except:
        labeled_iter = iter(labeled_train_loader)
        img_x, seg_map = next(labeled_iter)
    try:
        # Load unlabeled batch of weakly augmented images
        img_u_w = next(unlabeled_iter)
    except:
        unlabeled_iter = iter(unlabeled_train_loader)
        img_u_w = next(unlabeled_iter)

    # Initializes RandAugment with n random augmentations.
    # So, every batch will have different random augmentations.
    randaugment = RandAugmentMC(n=2, m=10)
    # Get strong transform to apply to both pseudo-label map and
    # weakly augmented image
    strong_transform = StrongAugmentation(
        randaugment=randaugment, mean=None, std=None
    )
    # Applies strong augmentation on weakly augmented images
    img_u_s = np.zeros((img_u_w.shape), dtype=np.float32)
    for i in range(img_u_w.shape[0]):
        img_u_w_i = img_u_w[i, :, :, :]
        img_u_w_i = img_u_w_i.cpu().detach().numpy()
        img_u_w_i = np.moveaxis(img_u_w_i, 0, -1)
        # Strongly-augmented image
        img_u_s_i = strong_transform(img_u_w_i)
        img_u_s[i, :, :, :] = img_u_s_i
    img_u_s = torch.from_numpy(img_u_s)
    # Moves data to device
    inputs = torch.cat((img_x, img_u_w, img_u_s)).to(device)
    seg_map = seg_map.to(device)
    optimizer.zero_grad()
    # Computes logits
    logits = model(inputs)
    logits_x = logits[:batch_size]
    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
    del logits
    # Supervised loss
    Lx = criterion(logits_x, seg_map)
    # Do not apply CutOut to the labels because the model has to
    # learn to interpolate when part of the image is missing.
    # It is only an augmentation on the inputs.
    randaugment.use_cutout(False)
    # Applies strong augmentation to pseudo label map
    tmp = np.zeros((logits_u_w.shape), dtype=np.float32)
    for i in range(logits_u_w.shape[0]):
        # When you debug visually: check that the strongly augmented
        # weak images correspond to the strongly augmented images
        # (e.g. the same ship should be at the same position in both
        # images).
        logits_u_w_i = logits_u_w[i, :, :, :]
        logits_u_w_i = logits_u_w_i.cpu().detach().numpy()
        logits_u_w_i = np.moveaxis(logits_u_w_i, 0, -1)
        logits_u_w_i = strong_transform(logits_u_w_i)
        tmp[i, :, :, :] = logits_u_w_i
    logits_u_w = tmp
    logits_u_s = logits_u_s.cpu().detach().numpy()
    # Sets all pixels that were added due to padding to a
    # constant value to later ignore them when computing the loss
    batch_size = logits_u_w.shape[0]
    num_categories = logits_u_w.shape[1]
    for idx_b in range(batch_size):
        for idx_cat in range(num_categories):
            # - logits_u_w_patch -> logits of the prediction of model
            #   on a weakly augmented image. Shape: (img_h, img_w)
            # - logits_u_s_patch -> logits of the prediction of model
            #   on a strongly augmented image. Shape: (img_h, img_w)
            logits_u_w_patch = logits_u_w[idx_b, idx_cat, :, :]
            logits_u_s_patch = logits_u_s[idx_b, idx_cat, :, :]
            logits_u_s_patch[
                np.where(logits_u_w_patch == padding_val)
            ] = IGNORE_INDEX
            logits_u_s_patch = torch.from_numpy(logits_u_s_patch)
            logits_u_s[idx_b, idx_cat, :, :] = logits_u_s_patch
    # Moves new logits to device
    # Weak-aug ones
    logits_u_w = torch.from_numpy(logits_u_w)
    logits_u_w = logits_u_w.to(device)
    # Strong-aug ones
    logits_u_s = torch.from_numpy(logits_u_s)
    logits_u_s = logits_u_s.to(device)
    # Applies softmax
    pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
    # target_u is the segmentation map containing the idx of the
    # class having the highest probability (for all pixels and for
    # all images in the batch)
    max_probs, targets_u = torch.max(pseudo_label, dim=classes_channel_idx)
    # Mask to ignore all pixels whose "confidence" is lower than
    # the specified threshold
    mask = max_probs.ge(threshold).float()
    # Mask to ignore all padding pixels
    padding_mask = logits_u_s[:, 0, :, :] == IGNORE_INDEX
    # Merge the two masks
    mask[padding_mask] = 0
    # Unsupervised loss
    # Multiplies the loss by the mask to ignore pixels
    loss_u = criterion_unsup(logits_u_s, targets_u) * torch.flatten(mask)
    if (loss_u).sum() == 0:
        Lu = 0.0
    else:
        Lu = (loss_u).sum() / torch.flatten(mask).sum()

    if Lu > 0:
        file_io.append(
            "./lu.txt",
            f"{model_name}. Lu: {str(Lu)}, epoch {epoch}, n_pixels: {len(mask[mask == 1])} \n",
        )
    print("-" * 20)
    print(f"Lx: {Lx}")
    print(f"Lu: {Lu}")
    print(f"Max prob: {max_probs.max()}")
    print(f"n_pixels: {len(mask[mask == 1])}")

    # Final loss
    loss = Lx + lambda_v * Lu
    loss.backward()

    # training_batches += logits_x.shape[0]  # TODO check
    training_loss.append((loss.data).tolist())

    optimizer.step()

    return loss, training_loss


def eval_step(
    image: torch.Tensor,
    target: torch.Tensor,
    criterion: nn.Module,
    test_loss: list[float],
    y_predicted: list[int],
    y_true: list[int],
    model: nn.Module,
    output_channels: int,
    device: torch.device,
) -> tuple[list[float], list[float]]:
    """Trains the model for one semi-supervised epoch.

    Args:
        image (torch.Tensor): image.
        target (torch.Tensor): segmentation map.
        criterion (nn.Module): supervised loss.
        test_loss (list[float]): list of semi-supervised test loss of batches.
        y_predicted (list[int]): list of predicted categories.
        y_true (list[int]): list of ground-truth categories.
        model (nn.Module): model.
        output_channels (int): number of output channels.
        device (torch.device): device.

    Returns:
        tuple[list[float], list[float]]: list of predictions, list of ground
          truths.
    """
    image = image.to(device)
    target = target.to(device)

    logits = model(image)

    loss = criterion(logits, target)

    # Accuracy metrics only on annotated pixels
    logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
    logits = logits.reshape((-1, output_channels))
    target = target.reshape(-1)
    mask = target != -1
    logits = logits[mask]
    target = target[mask]

    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
    target = target.cpu().numpy()

    test_loss.append((loss.data * target.shape[0]).tolist())
    y_predicted += probs.argmax(1).tolist()
    y_true += target.tolist()
    return y_predicted, y_true


def get_criterion(
    supervised: bool,
    alphas: list[float],
    device: torch.device,
    gamma: float = 2.0,
) -> nn.Module:
    """Gets the instance to compute the loss.

    Args:
        supervised (bool): True to get supervised criterion.
          False to return unsupervised criterion.
        alphas (list[float]): alpha coefficients of the Focal loss.
        device (torch.device): device.
        gamma (float): gamma coefficient of the Focal loss.

    Returns:
        nn.Module: the instance to compute the supervised loss.
    """
    if supervised:
        criterion = FocalLoss(
            alpha=alphas.to(device),
            gamma=gamma,
            reduction="mean",
            ignore_index=IGNORE_INDEX,
        )
    else:
        criterion = FocalLoss(
            alpha=alphas.to(device),
            gamma=gamma,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
    return criterion


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


def get_lr_scheduler(
    reduce_lr_on_plateau: int, optimizer: torch.optim, lr_steps: list[int]
) -> torch.optim.lr_scheduler:
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


def get_model(
    input_channels: int, output_channels: int, hidden_channels: int
) -> nn.Module:
    """Gets the model.

    Args:
        input_channels (int): num of input channels of the neural network.
        output_channels (int): num of output channels of the neural network.
        hidden_channels (int): num of hidden layers of the neural network.

    Returns:
        nn.Module: model (neural network).
    """
    model = UNet(
        input_bands=input_channels,
        output_classes=output_channels,
        hidden_channels=hidden_channels,
    )
    return model


def get_transform_train() -> transforms.Compose:
    """Gets the transformation to be applied to the training dataset.

    Returns:
        transforms.Compose: the transformation to be applied to the training set.
    """
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            DiscreteRandomRotation(ANGLES_FIXED_ROTATION),
            transforms.RandomHorizontalFlip(),
        ]
    )
    return transform_train


def get_transform_test() -> transforms.Compose:
    """Gets the transformation to be applied to the test dataset.

    Returns:
        transforms.Compose: the transformation to be applied to the test set.
    """
    transform_test = transforms.Compose([transforms.ToTensor()])
    return transform_test


def check_num_alphas(
    alphas: torch.Tensor,
    output_channels: int,
    aggregate_classes: CategoryAggregation,
) -> None:
    """Checks that the number of alpha coefficients is equal to the number of
    output channels.

    Args:
        alphas (torch.Tensor): alpha coefficients.
        output_channels (int): output channels.
        aggregate_classes (CategoryAggregation): type of aggregation of classes.

    Raises:
        Exception: exception raised if the number of alpha coefficients is not
        equal to the number of output channels.
    """
    if len(alphas) != output_channels:
        raise Exception(
            f"There should be as many alphas as the number of categories, which in this case is {output_channels} because the parameter aggregate_classes was set to {aggregate_classes}"
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
    """Gets the number of output channels.

    Args:
        aggregate_classes (CategoryAggregation): type of aggregation of
          classes.

    Raises:
        Exception: if the type of aggregation is not existent.

    Returns:
        int: the number of output channels.
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
    """Updates the path where to store checkpooints.

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
