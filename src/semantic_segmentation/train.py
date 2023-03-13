"""
Initial Implementation: Ioannis Kakogeorgiou
This modified implementation: Luca Marini
"""
import os
import ast
import sys
import json
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname as up
import datetime

import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils.assets import labels, labels_binary, labels_multi
from src.utils.utils import get_today_str
from src.semantic_segmentation.supervised.focal_loss import FocalLoss

from src.semantic_segmentation.supervised.models.unet import UNet
from src.semantic_segmentation.dataloader import (
    AnomalyMarineDataset,
    RandomRotationTransform,
    gen_weights,
    TrainMode,
    CategoryAggregation,
)

from src.utils.metrics import Evaluation
from src.utils.constants import CLASS_DISTR, BANDS_MEAN, BANDS_STD

root_path = up(up(up(os.path.abspath(__file__))))

logging.basicConfig(
    filename=os.path.join(root_path, "logs", "log_unet.log"),
    filemode="a",
    level=logging.INFO,
    format="%(name)s - %(levelname)s - %(message)s",
)
logging.info("*" * 10)


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


###############################################################
# Training                                                    #
###############################################################


def main(options):
    # Reproducibility
    # Limit the number of sources of nondeterministic behavior
    seed_all(0)
    g = torch.Generator()
    g.manual_seed(0)

    # Tensorboard
    writer = SummaryWriter(
        os.path.join(
            root_path, "logs", options["tensorboard"], options["today_str"]
        )
    )

    # Transformations

    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            RandomRotationTransform([-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip(),
        ]
    )

    transform_test = transforms.Compose([transforms.ToTensor()])
    class_distr = CLASS_DISTR
    standardization = transforms.Normalize(BANDS_MEAN, BANDS_STD)

    # Construct Data loader

    if options["mode"] == TrainMode.TRAIN.value:

        dataset_train = AnomalyMarineDataset(
            TrainMode.TRAIN.value,
            transform=transform_train,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
        )
        dataset_test = AnomalyMarineDataset(
            TrainMode.VAL.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
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
        # TODO: update (e.g. transformations and other)
        dataset_train = AnomalyMarineDataset(
            TrainMode.TRAIN_SSL.value,
            transform=transform_train,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            perc_labeled=options["perc_labeled"],
        )
        dataset_test = AnomalyMarineDataset(
            TrainMode.VAL.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
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

    elif options["mode"] == TrainMode.TEST.value:

        dataset_test = AnomalyMarineDataset(
            TrainMode.TEST.value,
            transform=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
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

    # Load model from specific epoch to continue the training or start the evaluation
    if options["resume_from_epoch"] > 1:

        resume_model_dir = os.path.join(
            options["checkpoint_path"], str(options["resume_from_epoch"])
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
        # - 'Sediment-Laden Water', 'Foam','Turbid Water', 'Shallow Water','Waves',
        #   'Cloud Shadows','Wakes', 'Mixed Water' with 'Marine Water'
        agg_distr_water = sum(class_distr_tmp[-9:])

        # Aggregate Distributions:
        # - 'Dense Sargassum','Sparse Sargassum' with 'Natural Organic Material'
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
        # Aggregate Distribution of all classes (except Marine Debris) with 'Others'
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
    alphas = 1 - class_distr
    # alphas = torch.Tensor(
    #    [0.25, 1]
    # )  # 0.25 * torch.ones_like(class_distr)  # 1 / class_distr
    # alphas = alphas / max(alphas)  # normalize
    criterion = FocalLoss(
        alpha=alphas.to(device),
        gamma=2.0,
        reduction="mean",
        ignore_index=-1,
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
            for (image, target) in tqdm(train_loader, desc="training"):

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
                    for (image, target) in tqdm(test_loader, desc="testing"):

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
                        test_loss.append(
                            (loss.data * target.shape[0]).tolist()
                        )
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
                        options["checkpoint_path"], str(epoch)
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
                            "Train loss": sum(training_loss)
                            / training_batches,
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
                    writer.add_scalar(
                        "F1/val weightF1", acc["weightF1"], epoch
                    )

                    writer.add_scalar("IoU/val MacroIoU", acc["IoU"], epoch)

                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()

    elif options["mode"] == TrainMode.TRAIN_SSL.value:
        # TODO
        pass
    # CODE ONLY FOR EVALUATION - TESTING MODE !
    elif options["mode"] == TrainMode.TEST.value:

        model.eval()

        test_loss = []
        test_batches = 0
        y_true = []
        y_predicted = []

        with torch.no_grad():
            for (image, target) in tqdm(test_loader, desc="testing"):

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

                probs = (
                    torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
                )
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
            logging.info(
                "Test loss was: " + str(sum(test_loss) / test_batches)
            )
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    today_str = get_today_str()

    # Options
    parser.add_argument(
        "--aggregate_classes",
        choices=list(CategoryAggregation),
        default=CategoryAggregation.BINARY.value,
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
            up(os.path.abspath(__file__)),
            "trained_models",
            today_str,
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

    args = parser.parse_args()
    args.today_str = today_str
    options = vars(args)  # convert to ordinary dict

    # lr_steps list or single float
    lr_steps = ast.literal_eval(options["lr_steps"])
    if type(lr_steps) is list:
        pass
    elif type(lr_steps) is int:
        lr_steps = [lr_steps]
    else:
        raise

    options["lr_steps"] = lr_steps

    logging.info("parsed input parameters:")
    logging.info(json.dumps(options, indent=2))
    main(options)
