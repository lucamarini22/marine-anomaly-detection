"""
Initial Implementation: Ioannis Kakogeorgiou
This modified implementation: Luca Marini
"""
import os
from enum import Enum
import ast
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from anomalymarinedetection.utils.assets import (
    labels,
    labels_binary,
    labels_multi,
)
from anomalymarinedetection.utils.string import get_today_str
from anomalymarinedetection.loss.focal_loss import (
    FocalLoss,
)
from anomalymarinedetection.models.unet import UNet
from anomalymarinedetection.dataset.dataloader import (
    AnomalyMarineDataset,
    DataLoaderType,
)
from anomalymarinedetection.dataset.transformations import (
    DiscreteRandomRotationTransform,
    TransformFixMatch,
    StrongAugmentation,
)
from anomalymarinedetection.utils.metrics import Evaluation
from anomalymarinedetection.utils.constants import (
    CLASS_DISTR,
    BANDS_MEAN,
    BANDS_STD,
    SEPARATOR,
)
from anomalymarinedetection.dataset.randaugment import (
    RandAugmentMC,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.utils.gen_weights import gen_weights
from anomalymarinedetection.dataset.get_labeled_and_unlabeled_rois import (
    get_labeled_and_unlabeled_rois,
)


class TrainMode(Enum):
    TRAIN = "SUP"
    TRAIN_SSL = "SSL"
    VAL = "val"
    TEST = "test"


def seed_all(seed):
    # Pytorch Reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    # DataLoader Workers Reproducibility
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def main(options):
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    model_name = (
        options["today_str"]
        + SEPARATOR
        + options["mode"]
        + SEPARATOR
        + options["aggregate_classes"]
    )

    # Tensorboard
    writer = SummaryWriter(
        os.path.join(
            # TODO: set log folder as an argument
            options["log_folder"],
            options["tensorboard"],
            model_name,
        )
    )

    # Transformations
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            DiscreteRandomRotationTransform([-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])
    class_distr = CLASS_DISTR
    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    # Construct Data loader
    if options["mode"] == TrainMode.TRAIN.value:
        dataset_train = AnomalyMarineDataset(
            DataLoaderType.TRAIN_SUP.value,
            transform=transform_train,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            path=options["dataset_path"],
        )
        dataset_test = AnomalyMarineDataset(
            DataLoaderType.VAL.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            path=options["dataset_path"],
        )

        train_loader = DataLoader(
            dataset_train,
            batch_size=options["batch"],
            shuffle=True,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=options["batch"],
            shuffle=False,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
        )
    elif options["mode"] == TrainMode.TRAIN_SSL.value:
        # Split training data into labeled and unlabeled sets
        ROIs, ROIs_u = get_labeled_and_unlabeled_rois(
            perc_labeled=options["perc_labeled"], path=options["dataset_path"]
        )

        # TODO: update (e.g. transformations and other)
        labeled_dataset_train = AnomalyMarineDataset(
            DataLoaderType.TRAIN_SUP.value,
            transform=transform_train,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            rois=ROIs,
            path=options["dataset_path"],
        )
        unlabeled_dataset_train = AnomalyMarineDataset(
            DataLoaderType.TRAIN_SSL.value,
            transform=TransformFixMatch(mean=BANDS_MEAN, std=BANDS_STD),
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            rois=ROIs_u,
            path=options["dataset_path"],
        )
        dataset_test = AnomalyMarineDataset(
            DataLoaderType.VAL.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            path=options["dataset_path"],
        )
        # TODO: fix batch size for labeled and unlabeled data loaders
        labeled_train_loader = DataLoader(
            labeled_dataset_train,
            batch_size=options["batch"],
            shuffle=True,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
        )
        unlabeled_train_loader = DataLoader(
            unlabeled_dataset_train,
            batch_size=options["batch"] * options["mu"],
            shuffle=True,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
            drop_last=True,
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=options["batch"],
            shuffle=False,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
        )

    elif options["mode"] == TrainMode.TEST.value:
        dataset_test = AnomalyMarineDataset(
            DataLoaderType.TEST.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            path=options["dataset_path"],
        )

        test_loader = DataLoader(
            dataset_test,
            batch_size=options["batch"],
            shuffle=False,
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            worker_init_fn=seed_worker,
            generator=g,
        )
    else:
        raise Exception("The mode option should be train, train_ssl, or test")

    if options["aggregate_classes"] == CategoryAggregation.MULTI.value:
        output_channels = len(labels_multi)
    elif options["aggregate_classes"] == CategoryAggregation.BINARY.value:
        output_channels = len(labels_binary)
    else:
        raise Exception(
            "The aggregated_classes option should be binary or multi"
        )

    # Use gpu or cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = UNet(
        input_bands=options["input_channels"],
        output_classes=output_channels,
        hidden_channels=options["hidden_channels"],
    )

    model.to(device)

    # Load model from specific epoch to continue the training or start the
    # evaluation
    if options["resume_from_epoch"] > 1:
        resume_model_dir = os.path.join(
            options["checkpoint_path"],
            model_name,
            str(options["resume_from_epoch"]),
        )
        model_file = os.path.join(resume_model_dir, "model.pth")
        logging.info("Loading model files from folder: %s" % model_file)

        checkpoint = torch.load(model_file, map_location=device)
        model.load_state_dict(checkpoint)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if options["aggregate_classes"] == CategoryAggregation.MULTI.value:
        # clone class_distrib tensor
        class_distr_tmp = class_distr.detach().clone()
        # Aggregate Distributions:
        # - 'Sediment-Laden Water', 'Foam','Turbid Water', 'Shallow Water',
        #   'Waves', 'Cloud Shadows','Wakes', 'Mixed Water' with 'Marine Water'
        agg_distr_water = sum(class_distr_tmp[-9:])

        # Aggregate Distributions:
        # - 'Dense Sargassum','Sparse Sargassum' with 'Natural Organic
        #    Material'
        agg_distr_algae_nom = sum(class_distr_tmp[1:4])

        agg_distr_ship = class_distr_tmp[labels.index("Ship")]
        agg_distr_cloud = class_distr_tmp[labels.index("Clouds")]

        class_distr[
            labels_multi.index("Algae/Natural Organic Material")
        ] = agg_distr_algae_nom
        class_distr[labels_multi.index("Marine Water")] = agg_distr_water

        class_distr[labels_multi.index("Ship")] = agg_distr_ship
        class_distr[labels_multi.index("Clouds")] = agg_distr_cloud

        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_multi)]

    elif options["aggregate_classes"] == CategoryAggregation.BINARY.value:
        # Aggregate Distribution of all classes (except Marine Debris) with
        # 'Others'
        agg_distr = sum(class_distr[1:])
        # Move the class distrib of Other to the 2nd position
        class_distr[labels_binary.index("Other")] = agg_distr
        # Drop class distribution of the aggregated classes
        class_distr = class_distr[: len(labels_binary)]

    # Weighted Cross Entropy Loss & adam optimizer
    # weight = gen_weights(class_distr, c=options["weight_param"])

    # criterion = torch.nn.CrossEntropyLoss(
    #    ignore_index=-1, reduction="mean", weight=weight.to(device)
    # )

    # TODO: modify class_distr when using ssl
    # (because you take a percentage of labels so the class distr of pixels
    # will change)
    # alphas = 1 - class_distr
    alphas = torch.Tensor(
        [0.50, 0.125, 0.125, 0.125, 0.125]
    )  # 0.25 * torch.ones_like(class_distr)  # 1 / class_distr
    # alphas = alphas / max(alphas)  # normalize
    criterion = FocalLoss(
        alpha=alphas.to(device),
        gamma=2.0,
        reduction="mean",
        ignore_index=-1,
    )

    criterion_unsup = FocalLoss(
        alpha=alphas.to(device),
        gamma=2.0,
        reduction="none",
        # ignore_index=-1
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=options["lr"], weight_decay=options["decay"]
    )

    # Learning Rate scheduler
    if options["reduce_lr_on_plateau"] == 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, options["lr_steps"], gamma=0.1, verbose=True
        )

    # Start training
    start = options["resume_from_epoch"] + 1
    epochs = options["epochs"]
    eval_every = options["eval_every"]

    # Write model-graph to Tensorboard
    if options["mode"] == TrainMode.TRAIN.value:
        dataiter = iter(train_loader)
        image_temp, _ = next(dataiter)
        writer.add_graph(model, image_temp.to(device))

        ###############################################################
        # Start Training                                              #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            print("_" * 40 + "Epoch " + str(epoch) + ": " + "_" * 40)
            training_loss = []
            training_batches = 0

            i_board = 0
            for image, target in tqdm(train_loader, desc="training"):
                image = image.to(device)
                target = target.to(device)

                optimizer.zero_grad()

                logits = model(image)

                loss = criterion(logits, target)

                loss.backward()

                training_batches += target.shape[0]

                training_loss.append((loss.data * target.shape[0]).tolist())

                optimizer.step()

                # Write running loss
                writer.add_scalar(
                    "training loss",
                    loss,
                    (epoch - 1) * len(train_loader) + i_board,
                )
                i_board += 1

            logging.info(
                "Training loss was: "
                + str(sum(training_loss) / training_batches)
            )

            ###############################################################
            # Start Evaluation                                            #
            ###############################################################

            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(test_loader, desc="testing"):
                        image = image.to(device)
                        target = target.to(device)

                        logits = model(image)

                        loss = criterion(logits, target)

                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(
                            logits, (0, 1, 2, 3), (0, 3, 1, 2)
                        )
                        logits = logits.reshape((-1, output_channels))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]

                        probs = (
                            torch.nn.functional.softmax(logits, dim=1)
                            .cpu()
                            .numpy()
                        )
                        target = target.cpu().numpy()

                        test_batches += target.shape[0]
                        test_loss.append((loss.data * target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist()
                        y_true += target.tolist()

                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)

                    ####################################################################
                    # Save Scores to the .log file and visualize also with tensorboard #
                    ####################################################################

                    acc = Evaluation(y_predicted, y_true)
                    logging.info("\n")
                    logging.info(
                        "Val loss was: " + str(sum(test_loss) / test_batches)
                    )
                    logging.info(
                        "STATISTICS AFTER EPOCH " + str(epoch) + ": \n"
                    )
                    logging.info("Evaluation: " + str(acc))

                    logging.info("Saving models")
                    model_dir = os.path.join(
                        options["checkpoint_path"],
                        model_name,
                        str(epoch),
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, "model.pth"),
                    )

                    writer.add_scalars(
                        "Loss per epoch",
                        {
                            "Val loss": sum(test_loss) / test_batches,
                            "Train loss": sum(training_loss) / training_batches,
                        },
                        epoch,
                    )

                    writer.add_scalar(
                        "Precision/val macroPrec", acc["macroPrec"], epoch
                    )
                    writer.add_scalar(
                        "Precision/val microPrec", acc["microPrec"], epoch
                    )
                    writer.add_scalar(
                        "Precision/val weightPrec", acc["weightPrec"], epoch
                    )

                    writer.add_scalar(
                        "Recall/val macroRec", acc["macroRec"], epoch
                    )
                    writer.add_scalar(
                        "Recall/val microRec", acc["microRec"], epoch
                    )
                    writer.add_scalar(
                        "Recall/val weightRec", acc["weightRec"], epoch
                    )

                    writer.add_scalar("F1/val macroF1", acc["macroF1"], epoch)
                    writer.add_scalar("F1/val microF1", acc["microF1"], epoch)
                    writer.add_scalar("F1/val weightF1", acc["weightF1"], epoch)

                    writer.add_scalar("IoU/val MacroIoU", acc["IoU"], epoch)

                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()

    elif options["mode"] == TrainMode.TRAIN_SSL.value:
        # TODO
        classes_channel_idx = 1

        labeled_iter = iter(labeled_train_loader)
        unlabeled_iter = iter(unlabeled_train_loader)

        # dataiter = iter(train_loader)
        # image_temp, _ = next(dataiter)
        # writer.add_graph(model, image_temp.to(device))

        ###############################################################
        # Start Training SEMI-SUPERVISED LEARINING                    #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            print("_" * 40 + "Epoch " + str(epoch) + ": " + "_" * 40)
            training_loss = []
            training_batches = 0

            i_board = 0
            for batch_idx in tqdm(range(len(labeled_iter)), desc="training"):
                try:
                    img_x, seg_map = next(labeled_iter)
                except:
                    labeled_iter = iter(labeled_train_loader)
                    img_x, seg_map = next(labeled_iter)
                try:
                    img_u_w = next(unlabeled_iter)
                except:
                    unlabeled_iter = iter(unlabeled_train_loader)
                    img_u_w = next(unlabeled_iter)

                # Initialize RandAugment with n random augmentations.
                # So, every batch will have different random augmentations.
                randaugment = RandAugmentMC(n=2, m=10)
                # Get strong transform to apply to both pseudo-label map and
                # weakly augmented image
                strong_transform = StrongAugmentation(
                    mean=BANDS_MEAN, std=BANDS_STD, randaugment=randaugment
                )
                # Applies strong augmentation on weakly augmented images
                img_u_s = np.zeros((img_u_w.shape), dtype=np.float32)
                for i in range(img_u_w.shape[0]):
                    img_u_w_i = img_u_w[i, :, :, :]
                    img_u_w_i = img_u_w_i.cpu().detach().numpy()
                    img_u_w_i = np.moveaxis(img_u_w_i, 0, -1)
                    a = img_u_w_i[:, :, 10]
                    b = img_u_w_i[:, :, 9]
                    img_u_s_i = strong_transform(img_u_w_i)
                    c = img_u_s_i[10, :, :]
                    d = img_u_s_i[9, :, :]
                    img_u_s[i, :, :, :] = img_u_s_i
                img_u_s = torch.from_numpy(img_u_s)
                # img_u_s = img_u_s.to(device)
                x = img_u_w[13, 4, :, :]

                z = img_u_s[13, 4, :, :]

                seg_map = seg_map.to(device)
                """ # DEBUGGING
                for i in range(img_x.shape[0]):
                    a = img_x[i, 7, :, :]
                    b = seg_map[i, :, :].float()

                    c = img_u_w[i, 7, :, :]
                    d = img_u_s[i, 7, :, :]
                    print()
                """

                # TODO: when deploying code to satellite hw, see if it's
                # faster to put everything to device and make one single
                # inference or to put one thing to device at a time and
                # make inference singularly
                # img = img.to(device)
                # seg_map = seg_map.to(device)

                # img_u_w = img_u_w.to(device)
                # img_u_s = img_u_s.to(device)

                inputs = torch.cat((img_x, img_u_w, img_u_s)).to(device)

                optimizer.zero_grad()

                logits = model(inputs)
                logits_x = logits[: options["batch"]]
                logits_u_w, logits_u_s = logits[options["batch"] :].chunk(2)
                del logits
                # print(randaugment.ops)
                # print(randaugment.probs_op)
                # print(randaugment.values_op)
                # Supervised loss
                Lx = criterion(logits_x, seg_map)

                # logits_u_s = model(img_u_s)

                # Do not apply CutOut to the labels because the model has to
                # learn to interpolate when part of the image is missing.
                # It is only an augmentation on the inputs.
                randaugment.use_cutout(False)
                # Applies strong augmentation to pseudo label map
                tmp = np.zeros((logits_u_w.shape), dtype=np.float32)
                for i in range(logits_u_w.shape[0]):
                    # print(randaugment.ops)
                    # print(randaugment.probs_op)
                    # print(randaugment.values_op)
                    logits_u_w_i = logits_u_w[i, :, :, :]
                    logits_u_w_i = logits_u_w_i.cpu().detach().numpy()
                    logits_u_w_i = np.moveaxis(logits_u_w_i, 0, -1)
                    # a = logits_u_w_i[:, :, 0]
                    # b = logits_u_w_i[:, :, 1]
                    logits_u_w_i = strong_transform(logits_u_w_i)
                    # c = logits_u_w_i[0, :, :]
                    # d = logits_u_w_i[1, :, :]  # TODO: visually debug these
                    tmp[i, :, :, :] = logits_u_w_i
                    # e = logits_u_s[i, 0, :, :]
                    # f = logits_u_s[i, 1, :, :]
                    # print()
                logits_u_w = torch.from_numpy(tmp)

                g = logits_u_w[13, 0, :, :]
                h = logits_u_w[13, 1, :, :]
                k = logits_u_s[13, 0, :, :]
                l = logits_u_s[13, 1, :, :]
                logits_u_w = logits_u_w.to(device)
                # logits_u_w = strong_transform(logits_u_w)
                pseudo_label = torch.softmax(
                    logits_u_w.detach(), dim=-1
                )  # / args.T, dim=-1) -> to add temperature
                # target_u contains the idx of the class having the highest
                # probability (for all pixels and for all images in the batch)
                max_probs, targets_u = torch.max(
                    pseudo_label, dim=classes_channel_idx
                )  # dim=-1)
                mask = max_probs.ge(options["threshold"]).float()
                # Unsupervised loss
                Lu = (
                    criterion_unsup(logits_u_s, targets_u) * torch.flatten(mask)
                ).mean()
                # Final loss
                loss = Lx + options["lambda"] * Lu
                loss.backward()

                # training_batches += logits_x.shape[0]  # TODO check

                training_loss.append((loss.data).tolist())  # TODO

                optimizer.step()

                # Write running loss
                writer.add_scalar(
                    "training loss",
                    loss,
                    (epoch - 1) * len(labeled_train_loader) + i_board,
                )
                i_board += 1

            # logging.info(
            #    "Training loss was: "
            #    + str(sum(training_loss) / training_batches)
            # )

            ###############################################################
            # Start Evaluation                                            #
            ###############################################################

            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(test_loader, desc="testing"):
                        image = image.to(device)
                        target = target.to(device)

                        logits = model(image)

                        loss = criterion(logits, target)

                        # Accuracy metrics only on annotated pixels
                        logits = torch.movedim(
                            logits, (0, 1, 2, 3), (0, 3, 1, 2)
                        )
                        logits = logits.reshape((-1, output_channels))
                        target = target.reshape(-1)
                        mask = target != -1
                        logits = logits[mask]
                        target = target[mask]

                        probs = (
                            torch.nn.functional.softmax(logits, dim=1)
                            .cpu()
                            .numpy()
                        )
                        target = target.cpu().numpy()

                        test_batches += target.shape[0]
                        test_loss.append((loss.data * target.shape[0]).tolist())
                        y_predicted += probs.argmax(1).tolist()
                        y_true += target.tolist()

                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)

                    ####################################################################
                    # Save Scores to the .log file and visualize also with tensorboard #
                    ####################################################################

                    acc = Evaluation(y_predicted, y_true)
                    logging.info("\n")
                    logging.info(
                        "Val loss was: " + str(sum(test_loss) / test_batches)
                    )
                    logging.info(
                        "STATISTICS AFTER EPOCH " + str(epoch) + ": \n"
                    )
                    logging.info("Evaluation: " + str(acc))

                    logging.info("Saving models")
                    model_dir = os.path.join(
                        options["checkpoint_path"],
                        model_name,
                        str(epoch),
                    )
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(
                        model.state_dict(),
                        os.path.join(model_dir, "model.pth"),
                    )

                    writer.add_scalars(
                        "Loss per epoch",
                        {
                            "Val loss": sum(test_loss) / test_batches,
                            "Train loss": np.mean(
                                training_loss
                            ),  # sum(training_loss)
                            #        / training_batches,
                        },
                        epoch,
                    )

                    writer.add_scalar(
                        "Precision/val macroPrec", acc["macroPrec"], epoch
                    )
                    writer.add_scalar(
                        "Precision/val microPrec", acc["microPrec"], epoch
                    )
                    writer.add_scalar(
                        "Precision/val weightPrec", acc["weightPrec"], epoch
                    )

                    writer.add_scalar(
                        "Recall/val macroRec", acc["macroRec"], epoch
                    )
                    writer.add_scalar(
                        "Recall/val microRec", acc["microRec"], epoch
                    )
                    writer.add_scalar(
                        "Recall/val weightRec", acc["weightRec"], epoch
                    )

                    writer.add_scalar("F1/val macroF1", acc["macroF1"], epoch)
                    writer.add_scalar("F1/val microF1", acc["microF1"], epoch)
                    writer.add_scalar("F1/val weightF1", acc["weightF1"], epoch)

                    writer.add_scalar("IoU/val MacroIoU", acc["IoU"], epoch)

                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()
    # CODE ONLY FOR EVALUATION - TESTING MODE !
    elif options["mode"] == TrainMode.TEST.value:
        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []

        with torch.no_grad():
            for image, target in tqdm(test_loader, desc="testing"):
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

                test_batches += target.shape[0]
                test_loss.append((loss.data * target.shape[0]).tolist())
                y_predicted += probs.argmax(1).tolist()
                y_true += target.tolist()

            y_predicted = np.asarray(y_predicted)
            y_true = np.asarray(y_true)

            ####################################################################
            # Save Scores to the .log file                                     #
            ####################################################################
            acc = Evaluation(y_predicted, y_true)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    today_str = get_today_str()

    # Options
    parser.add_argument(
        "--aggregate_classes",
        choices=list(CategoryAggregation),
        default=CategoryAggregation.MULTI.value,
        type=str,
        help="Aggregate classes into:\
            multi (Marine Water, Algae/OrganicMaterial, Marine Debris, Ship, and Cloud);\
                binary (Marine Debris and Other); \
                    no (keep the original 15 classes)",
    )  # TODO: re-implement the option to keep the 15 original classes
    parser.add_argument(
        "--mode",
        choices=list(TrainMode),
        default=TrainMode.TRAIN_SSL.value,
        help="Mode",
    )
    ###### SSL hyperparameters ######
    parser.add_argument(
        "--perc_labeled",
        default=0.1,
        help=(
            "Percentage of labeled training set. This argument has "
            "effect only when --mode=TrainMode.TRAIN_SSL.value. "
            " The percentage of the unlabeled training set will be "
            " 1 - perc_labeled."
        ),
        type=float,
    )
    parser.add_argument(
        "--mu",
        default=9,
        help=("Unlabeled data ratio."),
        type=float,
    )
    parser.add_argument(
        "--threshold",
        default=0.9,
        help=("Confidence threshold for pseudo-labels."),
        type=float,
    )
    parser.add_argument(
        "--lambda",
        default=1,
        type=float,
        help="Coefficient of unlabeled loss.",
    )
    ####################################
    parser.add_argument(
        "--epochs",
        default=20000,
        type=int,
        help="Number of epochs to run",  # 45
    )
    parser.add_argument("--batch", default=5, type=int, help="Batch size")
    parser.add_argument(
        "--resume_from_epoch",
        default=0,
        type=int,
        help="Load model from previous epoch",
    )

    parser.add_argument(
        "--input_channels", default=11, type=int, help="Number of input bands"
    )

    parser.add_argument(
        "--hidden_channels",
        default=16,
        type=int,
        help="Number of hidden features",
    )
    parser.add_argument(
        "--dataset_path", help="path of dataset", default="data"
    )
    # parser.add_argument(
    #    "--weight_param",
    #    default=1.03,
    #    type=float,
    #    help="Weighting parameter for Loss Function",
    # )

    # Optimization
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument(
        "--decay", default=0, type=float, help="Learning rate decay"
    )
    parser.add_argument(
        "--reduce_lr_on_plateau",
        default=0,
        type=int,
        help="Reduce learning rate when no increase (0 or 1)",
    )
    parser.add_argument(
        "--lr_steps",
        default="[40]",
        type=str,
        help="Specify the steps that the lr will be reduced",
    )

    # Evaluation/Checkpointing
    parser.add_argument(
        "--checkpoint_path",
        default=os.path.join(
            "results",
            "trained_models",
        ),
        help="Folder to save checkpoints into (empty = this folder)",
    )
    parser.add_argument(
        "--eval_every",
        default=1,
        type=int,
        help="How frequently to run evaluation (epochs)",
    )

    # misc
    parser.add_argument(
        "--num_workers",
        default=1,
        type=int,
        help="How many cpus for loading data (0 is the main process)",
    )
    parser.add_argument(
        "--pin_memory",
        default=False,
        type=bool,
        help="Use pinned memory or not",
    )
    parser.add_argument(
        "--prefetch_factor",
        default=1,
        type=int,
        help="Number of sample loaded in advance by each worker",
    )
    parser.add_argument(
        "--persistent_workers",
        default=True,
        type=bool,
        help="This allows to maintain the workers Dataset instances alive.",
    )
    parser.add_argument(
        "--tensorboard",
        default="tsboard_segm",
        type=str,
        help="Name for tensorboard run",
    )
    parser.add_argument(
        "--log_folder",
        default="logs",
        type=str,
        help="Path of the log folder",
    )

    args = parser.parse_args()
    args.today_str = today_str
    # convert to ordinary dict
    options = vars(args)

    if options["mode"] == TrainMode.TRAIN_SSL.value:
        options["checkpoint_path"] = os.path.join(
            options["checkpoint_path"], "semi-supervised"
        )
    elif options["mode"] == TrainMode.TRAIN.value:
        options["checkpoint_path"] = os.path.join(
            options["checkpoint_path"], "supervised"
        )
    else:
        pass

    if not os.path.isdir(options["checkpoint_path"]):
        raise Exception(
            f'The checkpoint directory {options["checkpoint_path"]} does not exist'
        )

    # lr_steps list or single float
    lr_steps = ast.literal_eval(options["lr_steps"])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise

    options["lr_steps"] = lr_steps
    # Logging
    logging.basicConfig(
        filename=os.path.join(options["log_folder"], "log_unet.log"),
        filemode="a",
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.info("*" * 10)

    logging.info("parsed input parameters:")
    logging.info(json.dumps(options, indent=2))
    main(options)
