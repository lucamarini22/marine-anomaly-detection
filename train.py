import os
import json
import logging
import numpy as np
from tqdm import tqdm
import torch

from anomalymarinedetection.dataset.get_dataloaders import (
    get_dataloaders_supervised,
    get_dataloaders_ssl,
)
from anomalymarinedetection.utils.metrics import Evaluation
from anomalymarinedetection.utils.constants import (
    CLASS_DISTR,
    BANDS_MEAN,
    BANDS_STD,
    SEPARATOR,
    PADDING_VAL,
)
from anomalymarinedetection.io.file_io import FileIO
from anomalymarinedetection.io.tbwriter import TBWriter
from anomalymarinedetection.trainmode import TrainMode
from anomalymarinedetection.parse_args_train import parse_args_train
from anomalymarinedetection.io.model_handler import (
    load_model,
    save_model,
    get_model_name,
)
from anomalymarinedetection.utils.seed import set_seed, set_seed_worker
from anomalymarinedetection.dataset.update_class_distribution import (
    update_class_distribution,
)
from anomalymarinedetection.utils.device import get_device, empty_cache
from anomalymarinedetection.utils.train_functions import (
    train_step_supervised,
    train_step_semi_supervised,
    eval_step,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
    get_model,
    get_transform_train,
    get_transform_test,
    check_num_alphas,
    get_output_channels,
    get_lr_steps,
    update_checkpoint_path,
    check_checkpoint_path_exist,
)


def main(options):
    # Use gpu or cpu
    device = get_device()
    file_io = FileIO()
    # Reproducibility
    seed = options["seed"]
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

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
    transform_train = get_transform_train()
    transform_test = get_transform_test()
    # TODO: modify class_distr when using ssl
    # (because you take a percentage of labels so the class distr of pixels
    # will change)
    class_distr = None  # CLASS_DISTR
    standardization = None  # transforms.Normalize(BANDS_MEAN, BANDS_STD)
    # Construct Data loader
    if options["mode"] == TrainMode.TRAIN:
        train_loader, test_loader = get_dataloaders_supervised(
            dataset_path=options["dataset_path"],
            transform_train=transform_train,
            transform_test=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            batch=options["batch"],
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            seed_worker_fn=set_seed_worker,
            generator=g,
        )
    elif options["mode"] == TrainMode.TRAIN_SSL:
        (
            labeled_train_loader,
            unlabeled_train_loader,
            test_loader,
        ) = get_dataloaders_ssl(
            dataset_path=options["dataset_path"],
            transform_train=transform_train,
            transform_test=transform_test,
            standardization=standardization,
            aggregate_classes=options["aggregate_classes"],
            batch=options["batch"],
            num_workers=options["num_workers"],
            pin_memory=options["pin_memory"],
            prefetch_factor=options["prefetch_factor"],
            persistent_workers=options["persistent_workers"],
            seed_worker_fn=set_seed_worker,
            generator=g,
            perc_labeled=options["perc_labeled"],
            mu=options["mu"],
            drop_last=True,
        )
    else:
        raise Exception("The mode option should be train, train_ssl, or test")

    output_channels = get_output_channels(options["aggregate_classes"])
    # Init model
    model = get_model(
        input_channels=options["input_channels"],
        output_channels=output_channels,
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
        empty_cache()

        start = int(options["resume_model"].split("/")[-2]) + 1
    else:
        start = 1

    if class_distr is not None:
        class_distr = update_class_distribution(
            options["aggregate_classes"], class_distr
        )
    # Coefficients of Focal loss
    alphas = torch.Tensor(options["alphas"])
    check_num_alphas(alphas, output_channels, options["aggregate_classes"])
    # Init of supervised loss
    criterion = get_criterion(
        supervised=True, alphas=alphas, device=device, gamma=options["gamma"]
    )
    if options["mode"] == TrainMode.TRAIN_SSL:
        # Init of unsupervised loss
        criterion_unsup = get_criterion(
            supervised=False,
            alphas=alphas,
            device=device,
            gamma=options["gamma"],
        )
    # Optimizer
    optimizer = get_optimizer(
        model, lr=options["lr"], weight_decay=options["decay"]
    )
    # Learning Rate scheduler
    scheduler = get_lr_scheduler(
        options["reduce_lr_on_plateau"], optimizer, options["lr_steps"]
    )
    # Start training
    epochs = options["epochs"]
    eval_every = options["eval_every"]

    if options["mode"] == TrainMode.TRAIN:
        dataiter = iter(train_loader)
        image_temp, _ = next(dataiter)
        # Write model-graph to Tensorboard
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
                loss, training_loss = train_step_supervised(
                    image,
                    target,
                    criterion,
                    training_loss,
                    model,
                    optimizer,
                    device,
                )
                training_batches += target.shape[0]
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

            # Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(test_loader, desc="testing"):
                        y_predicted, y_true = eval_step(
                            image,
                            target,
                            criterion,
                            test_loss,
                            y_predicted,
                            y_true,
                            model,
                            output_channels,
                            device,
                        )
                        test_batches += target.shape[0]
                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    # Save Scores to the .log file and visualize also with tensorboard
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

    elif options["mode"] == TrainMode.TRAIN_SSL:
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
                loss, training_loss = train_step_semi_supervised(
                    file_io,
                    labeled_train_loader,
                    unlabeled_train_loader,
                    labeled_iter,
                    unlabeled_iter,
                    criterion,
                    criterion_unsup,
                    training_loss,
                    model,
                    model_name,
                    optimizer,
                    epoch,
                    device,
                    options["batch"],
                    classes_channel_idx,
                    options["threshold"],
                    options["lambda"],
                    PADDING_VAL,
                )
                # Write running loss
                tb_writer.add_scalar(
                    "training loss",
                    loss,
                    (epoch - 1) * len(labeled_train_loader) + i_board,
                )
                i_board += 1

            logging.info("Training loss was: " + str(np.mean(training_loss)))

            # Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                test_loss = []
                test_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(test_loader, desc="testing"):
                        y_predicted, y_true = eval_step(
                            image,
                            target,
                            criterion,
                            test_loss,
                            y_predicted,
                            y_true,
                            model,
                            output_channels,
                            device,
                        )
                        test_batches += target.shape[0]

                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    # Save Scores to the .log file and visualize also with tensorboard
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
                            "Train loss": np.mean(training_loss),
                        },
                        epoch,
                    )
                    tb_writer.add_eval_metrics(acc, epoch)

                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(sum(test_loss) / test_batches)
                else:
                    scheduler.step()

                model.train()


if __name__ == "__main__":
    options = parse_args_train()
    options["checkpoint_path"] = update_checkpoint_path(
        options["mode"], options["checkpoint_path"]
    )
    check_checkpoint_path_exist(options["checkpoint_path"])
    options["lr_steps"] = get_lr_steps(options["lr_steps"])
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
