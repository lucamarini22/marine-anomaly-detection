import os
import ast
import json
import random
import logging
import numpy as np
from tqdm import tqdm
from os.path import dirname as up

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from anomalymarinedetection.utils.assets import (
    labels,
    labels_binary,
    labels_multi,
)
from anomalymarinedetection.loss.focal_loss import FocalLoss
from anomalymarinedetection.models.unet import UNet
from anomalymarinedetection.dataset.anomalymarinedataset import (
    AnomalyMarineDataset,
)
from anomalymarinedetection.dataset.augmentation.weakaugmentation import (
    WeakAugmentation,
)
from anomalymarinedetection.dataset.augmentation.strongaugmentation import (
    StrongAugmentation,
)
from anomalymarinedetection.dataset.augmentation.randaugment import (
    RandAugmentMC,
)
from anomalymarinedetection.dataset.augmentation.discreterandomrotation import (
    DiscreteRandomRotation,
)
from anomalymarinedetection.utils.metrics import Evaluation
from anomalymarinedetection.utils.constants import (
    CLASS_DISTR,
    BANDS_MEAN,
    BANDS_STD,
    SEPARATOR,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
from anomalymarinedetection.utils.gen_weights import gen_weights
from anomalymarinedetection.dataset.get_labeled_and_unlabeled_rois import (
    get_labeled_and_unlabeled_rois,
)
from anomalymarinedetection.io.file_io import FileIO
from anomalymarinedetection.io.tbwriter import TBWriter
from anomalymarinedetection.trainmode import TrainMode
from anomalymarinedetection.parse_args import parse_args
from anomalymarinedetection.io.model_handler import (
    load_model,
    save_model,
    get_model_name,
)


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
    file_io = FileIO()
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    model_name = get_model_name(
        options["resume_model"],
        options["mode"],
        options["aggregate_classes"],
        options["today_str"],
        SEPARATOR,
    )

    # Tensorboard
    tb_writer = TBWriter(
        os.path.join(
            options["log_folder"],
            options["tensorboard"],
            model_name,
        )
    )

    # Transformations
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            DiscreteRandomRotation([-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])
    # class_distr = CLASS_DISTR
    standardization = None  # transforms.Normalize(BANDS_MEAN, BANDS_STD)

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
            transform=WeakAugmentation(
                mean=None, std=None
            ),  # WeakAugmentation(mean=BANDS_MEAN, std=BANDS_STD),
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
    if options["resume_model"] is not None:
        logging.info(
            f"Loading model files from folder: {options['resume_model']}"
        )
        load_model(model, options["resume_model"], device)

        del checkpoint  # dereference
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = int(options["resume_model"].split("/")[-2]) + 1
    else:
        start = 1
    """ # Commented because I'm not using class_distr atm
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
    """

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
    epochs = options["epochs"]
    eval_every = options["eval_every"]

    # Write model-graph to Tensorboard
    if options["mode"] == TrainMode.TRAIN.value:
        dataiter = iter(train_loader)
        image_temp, _ = next(dataiter)
        tb_writer.add_graph(model, image_temp.to(device))

        ###############################################################
        # Start Supervised Training                                   #
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
                tb_writer.add_scalar(
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
                    save_model(model, os.path.join(model_dir, "model.pth"))

                    tb_writer.add_scalars(
                        "Loss per epoch",
                        {
                            "Val loss": sum(test_loss) / test_batches,
                            "Train loss": sum(training_loss) / training_batches,
                        },
                        epoch,
                    )
                    tb_writer.add_eval_metrics(acc, epoch)

                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()

    elif options["mode"] == TrainMode.TRAIN_SSL.value:
        classes_channel_idx = 1

        labeled_iter = iter(labeled_train_loader)
        unlabeled_iter = iter(unlabeled_train_loader)

        ###############################################################
        # Start SEMI-SUPERVISED LEARNING Training                     #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            print("_" * 40 + "Epoch " + str(epoch) + ": " + "_" * 40)
            training_loss = []
            training_batches = 0

            i_board = 0
            for _ in tqdm(range(len(labeled_iter)), desc="training"):
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
                strong_transform = StrongAugmentation(randaugment=randaugment)
                # mean=BANDS_MEAN, std=BANDS_STD, randaugment=randaugment)
                # Applies strong augmentation on weakly augmented images
                img_u_s = np.zeros((img_u_w.shape), dtype=np.float32)
                for i in range(img_u_w.shape[0]):
                    img_u_w_i = img_u_w[i, :, :, :]
                    img_u_w_i = img_u_w_i.cpu().detach().numpy()
                    img_u_w_i = np.moveaxis(img_u_w_i, 0, -1)
                    # Strongly-augmented image
                    img_u_s_i = strong_transform(img_u_w_i)
                    # a = img_u_w_i[:, :, 10]
                    # b = img_u_w_i[:, :, 9]

                    # c = img_u_s_i[10, :, :]
                    # d = img_u_s_i[9, :, :]
                    img_u_s[i, :, :, :] = img_u_s_i
                img_u_s = torch.from_numpy(img_u_s)
                seg_map = seg_map.to(device)
                """ DEBUGGING
                for i in range(img_x.shape[0]):
                    a = img_x[i, 8, :, :]
                    b = seg_map[i, :, :].float()

                    c = img_u_w[i, 8, :, :]
                    d = img_u_s[i, 8, :, :]
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

                # Supervised loss
                Lx = criterion(logits_x, seg_map)
                # Do not apply CutOut to the labels because the model has to
                # learn to interpolate when part of the image is missing.
                # It is only an augmentation on the inputs.
                randaugment.use_cutout(False)
                # Applies strong augmentation to pseudo label map
                tmp = np.zeros((logits_u_w.shape), dtype=np.float32)
                for i in range(logits_u_w.shape[0]):
                    logits_u_w_i = logits_u_w[i, :, :, :]
                    logits_u_w_i = logits_u_w_i.cpu().detach().numpy()
                    logits_u_w_i = np.moveaxis(logits_u_w_i, 0, -1)
                    # a = logits_u_w_i[:, :, 0]
                    # b = logits_u_w_i[:, :, 1]
                    # min_logits_u_w_i, max_logits_u_w_i = (
                    #    logits_u_w_i.min(),
                    #    logits_u_w_i.max(),
                    # )
                    # logits_u_w_i = normalize_img(
                    #    logits_u_w_i, min_logits_u_w_i, max_logits_u_w_i
                    # )
                    c = logits_u_w_i[:, :, 0]
                    d = logits_u_w_i[:, :, 1]
                    logits_u_w_i = strong_transform(logits_u_w_i)
                    e = logits_u_w_i[0, :, :]
                    f = logits_u_w_i[1, :, :]  # TODO: visually debug these
                    tmp[i, :, :, :] = logits_u_w_i
                    g = logits_u_s[i, 0, :, :]
                    h = logits_u_s[i, 1, :, :]

                    # aa = img_x[i, 7, :, :]
                    # bb = seg_map[i, :, :].float()

                    # cc = img_u_w[i, 4, :, :]
                    # dd = img_u_s[i, 4, :, :]
                    print()
                logits_u_w = torch.from_numpy(tmp)

                logits_u_w = logits_u_w.to(device)

                pseudo_label = torch.softmax(
                    logits_u_w.detach(), dim=-1
                )  # / args.T, dim=-1) -> to add temperature
                # target_u contains the idx of the class having the highest
                # probability (for all pixels and for all images in the batch)
                max_probs, targets_u = torch.max(
                    pseudo_label, dim=classes_channel_idx
                )  # dim=-1)
                # aaa = torch.max(logits_u_s, dim=classes_channel_idx)[1]
                mask = max_probs.ge(options["threshold"]).float()
                # Unsupervised loss
                Lu = (
                    criterion_unsup(logits_u_s, targets_u) * torch.flatten(mask)
                ).mean()

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
                loss = Lx + options["lambda"] * Lu
                loss.backward()

                # training_batches += logits_x.shape[0]  # TODO check

                training_loss.append((loss.data).tolist())  # TODO

                optimizer.step()

                # Write running loss
                tb_writer.add_scalar(
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
                    save_model(model, os.path.join(model_dir, "model.pth"))

                    tb_writer.add_scalars(
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
                    tb_writer.add_eval_metrics(acc, epoch)

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
    options = parse_args()

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
