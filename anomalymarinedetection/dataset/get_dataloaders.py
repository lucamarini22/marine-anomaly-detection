from torch.utils.data import DataLoader

from anomalymarinedetection.dataset.anomalymarinedataset import (
    AnomalyMarineDataset,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
from anomalymarinedetection.dataset.augmentation.weakaugmentation import (
    WeakAugmentation,
)
from anomalymarinedetection.dataset.get_labeled_and_unlabeled_rois import (
    get_labeled_and_unlabeled_rois,
)


def get_dataloaders_supervised(
    dataset_path,
    transform_train,
    transform_test,
    standardization,
    aggregate_classes,
    batch,
    num_workers,
    pin_memory,
    prefetch_factor,
    persistent_workers,
    seed_worker_fn,
    generator,
):
    dataset_train = AnomalyMarineDataset(
        DataLoaderType.TRAIN_SUP,
        transform=transform_train,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )
    dataset_test = AnomalyMarineDataset(
        DataLoaderType.VAL,
        transform=transform_test,
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

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return train_loader, test_loader


def get_dataloaders_ssl(
    dataset_path,
    transform_train,
    transform_test,
    standardization,
    aggregate_classes,
    batch,
    num_workers,
    pin_memory,
    prefetch_factor,
    persistent_workers,
    seed_worker_fn,
    generator,
    perc_labeled,
    mu,
    drop_last=True,
):
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
        transform=WeakAugmentation(
            mean=None, std=None
        ),
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        rois=ROIs_u,
        path=dataset_path,
    )
    dataset_test = AnomalyMarineDataset(
        DataLoaderType.VAL,
        transform=transform_test,
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

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return labeled_train_loader, unlabeled_train_loader, test_loader


def get_dataloaders_eval(
    dataset_path,
    transform_test,
    standardization,
    aggregate_classes,
    batch,
    num_workers,
    pin_memory,
    prefetch_factor,
    persistent_workers,
    seed_worker_fn,
    generator,
):
    dataset_test = AnomalyMarineDataset(
        DataLoaderType.TEST,
        transform=transform_test,
        standardization=standardization,
        aggregate_classes=aggregate_classes,
        path=dataset_path,
    )

    test_loader = DataLoader(
        dataset_test,
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker_fn,
        generator=generator,
    )
    return test_loader
