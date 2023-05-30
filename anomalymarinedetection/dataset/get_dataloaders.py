from typing import Callable
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


from anomalymarinedetection.dataset.anomalymarinedataset import (
    AnomalyMarineDataset,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
from anomalymarinedetection.dataset.augmentation.weakaugmentation import (
    WeakAugmentation,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.dataset.get_labeled_and_unlabeled_rois import (
    get_labeled_and_unlabeled_rois,
)


def get_dataloaders_supervised(
    dataset_path: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    standardization: transforms.Normalize,
    aggregate_classes: CategoryAggregation,
    batch: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed_worker_fn: Callable,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Gets the dataloaders for supervised training.

    Args:
        dataset_path (str): path of the dataset.
        transform_train (transforms.Compose): transformations to be applied
          to training set.
        transform_val (transforms.Compose): transformations to be applied
          to the validation set.
        standardization (transforms.Normalize): standardization.
        aggregate_classes (CategoryAggregation): type of classes aggregation.
        batch (int): size of batch.
        num_workers (int, optional): how many subprocesses to use for data
          loading. ``0`` means that the data will be loaded in the main
          process.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into
          device/CUDA pinned memory before returning them.
        prefetch_factor (int): Number of batches loaded in advance by each
          worker.
        persistent_workers (bool): If ``True``, the data loader will not
          shutdown the worker processes after a dataset has been consumed
          once. This allows to maintain the workers `Dataset` instances alive.
        seed_worker_fn (Callable): If not ``None``, this will be called on
          each worker subprocess with the worker id (an int in
          ``[0, num_workers - 1]``) as input, after seeding and before data
          loading.
        generator (torch.Generator): If not ``None``, this RNG will be used
          by RandomSampler to generate random indexes and multiprocessing to
          generate `base_seed` for workers.

    Returns:
        tuple[DataLoader, DataLoader]: training and validation dataloaders.
    """
    dataset_train = AnomalyMarineDataset(
        DataLoaderType.TRAIN_SUP,
        transform=transform_train,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )
    dataset_val = AnomalyMarineDataset(
        DataLoaderType.VAL,
        transform=transform_val,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return train_loader, val_loader


def get_dataloaders_ssl_single_train_set(
    dataset_path: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    weakly_transform:transforms.Compose,
    standardization: transforms.Normalize,
    aggregate_classes: CategoryAggregation,
    batch: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed_worker_fn: Callable,
    generator: torch.Generator,
) -> tuple[DataLoader, DataLoader]:
    """Gets dataloaders to perform semi-supervised learning with one unique 
    training set. Having a patch of the training set:
      - its labeled pixels will be used to compute the supervised loss.
      - its unlabeled pixels will be used to compute the unsupervised loss.

    Args:
        dataset_path (str): path of the dataset.
        transform_train (transforms.Compose): transformations to be applied
          to training set for the supervised loss.
        transform_val (transforms.Compose): transformations to be applied
          to the validation set.
        weakly_transform (transforms.Compose): transformation to be applied to
          the training set for the unsupervised loss.
        standardization (transforms.Normalize): standardization.
        aggregate_classes (CategoryAggregation): type of classes aggregation.
        batch (int): size of batch.
        num_workers (int, optional): how many subprocesses to use for data
          loading. ``0`` means that the data will be loaded in the main
          process.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into
          device/CUDA pinned memory before returning them.
        prefetch_factor (int): Number of batches loaded in advance by each
          worker.
        persistent_workers (bool): If ``True``, the data loader will not
          shutdown the worker processes after a dataset has been consumed
          once. This allows to maintain the workers `Dataset` instances alive.
        seed_worker_fn (Callable): If not ``None``, this will be called on
          each worker subprocess with the worker id (an int in
          ``[0, num_workers - 1]``) as input, after seeding and before data
          loading.
        generator (torch.Generator): If not ``None``, this RNG will be used
          by RandomSampler to generate random indexes and multiprocessing to
          generate `base_seed` for workers.

    Returns:
        tuple[DataLoader, DataLoader]: training and validation dataloaders.
    """
    dataset_train = AnomalyMarineDataset(
        DataLoaderType.TRAIN_SSL_SUP,
        transform=transform_train,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
        second_transform=weakly_transform
    )
    dataset_val = AnomalyMarineDataset(
        DataLoaderType.VAL,
        transform=transform_val,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )

    train_loader = DataLoader(
        dataset_train,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return train_loader, val_loader


def get_dataloaders_ssl_separate_train_sets(
    dataset_path: str,
    transform_train: transforms.Compose,
    transform_val: transforms.Compose,
    standardization: transforms.Normalize,
    aggregate_classes: CategoryAggregation,
    batch: int,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    persistent_workers: bool,
    seed_worker_fn: Callable,
    generator: torch.Generator,
    perc_labeled: float,
    mu: int,
    drop_last: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Gets dataloaders to perform semi-supervised learning with two different
    training sets:
      - D_s: weakly-labeled dataset
      - D_u: unlabeled dataset
    (D_s U D_u = D, i.e. the union of the 2 datasets is the full training 
    dataset).
    So:
     - Patches in D_s are used to compute the supervised loss.
     - Patches in D_u are used to compute the unsupervised loss.
     
    Args:
        dataset_path (str): path of the dataset.
        transform_train (transforms.Compose): transformations to be applied
          to training set.
        transform_val (transforms.Compose): transformations to be applied
          to the validation set.
        standardization (transforms.Normalize): standardization.
        aggregate_classes (CategoryAggregation): type of classes aggregation.
        batch (int): size of batch.
        num_workers (int, optional): how many subprocesses to use for data
          loading. ``0`` means that the data will be loaded in the main
          process.
        pin_memory (bool): If ``True``, the data loader will copy Tensors into
          device/CUDA pinned memory before returning them.
        prefetch_factor (int): Number of batches loaded in advance by each
          worker.
        persistent_workers (bool): If ``True``, the data loader will not
          shutdown the worker processes after a dataset has been consumed
          once. This allows to maintain the workers `Dataset` instances alive.
        seed_worker_fn (Callable): If not ``None``, this will be called on
          each worker subprocess with the worker id (an int in
          ``[0, num_workers - 1]``) as input, after seeding and before data
          loading.
        generator (torch.Generator): If not ``None``, this RNG will be used
          by RandomSampler to generate random indexes and multiprocessing to
          generate `base_seed` for workers.
        perc_labeled (float): Percentage of labeled training set.
        mu (int): Unlabeled data ratio.
        drop_last (bool, optional): set to True to drop the last incomplete
          batch, if the dataset size is not divisible by the batch size.
          If False and the size of dataset is not divisible by the batch size,
          then the last batch will be smaller. Defaults to True.

    Returns:
        tuple[DataLoader, DataLoader, DataLoader]: labeled training dataloader,
        unlabeled training dataloader, and validation dataloader.
    """
    # Split training data into labeled and unlabeled sets
    ROIs, ROIs_u = get_labeled_and_unlabeled_rois(
        perc_labeled=perc_labeled, path=dataset_path
    )

    # TODO: update (e.g. transformations and other)
    labeled_dataset_train = AnomalyMarineDataset(
        DataLoaderType.TRAIN_SUP,
        transform=transform_train,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        rois=ROIs,
        path=dataset_path,
        perc_labeled=perc_labeled,
    )
    unlabeled_dataset_train = AnomalyMarineDataset(
        DataLoaderType.TRAIN_SSL,
        transform=WeakAugmentation(mean=None, std=None),
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        rois=ROIs_u,
        path=dataset_path,
    )
    dataset_val = AnomalyMarineDataset(
        DataLoaderType.VAL,
        transform=transform_val,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )
    # TODO: fix batch size for labeled and unlabeled data loaders
    labeled_train_loader = DataLoader(
        labeled_dataset_train,
        batch_size=batch,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
        drop_last=drop_last,
    )
    unlabeled_train_loader = DataLoader(
        unlabeled_dataset_train,
        batch_size=batch * mu,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
        drop_last=drop_last,
    )

    val_loader = DataLoader(
        dataset_val,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return labeled_train_loader, unlabeled_train_loader, val_loader
