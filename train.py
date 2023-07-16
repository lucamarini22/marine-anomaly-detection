import sys
import os
import json
from loguru import logger
import logging
import numpy as np
from tqdm import tqdm
import torch

from marineanomalydetection.dataset.get_dataloaders import (
    get_dataloaders_supervised,
    get_dataloaders_ssl_separate_train_sets,
    get_dataloaders_ssl_single_train_set,
)
from marineanomalydetection.utils.metrics import Evaluation
from marineanomalydetection.utils.constants import (
    CLASS_DISTR,
    BANDS_MEAN,
    BANDS_STD,
    SEPARATOR,
    PADDING_VAL,
    LOG_SET,
    LOG_STD_OUT,
    LOG_SSL_LOSS,
)
from marineanomalydetection.io.file_io import FileIO
from marineanomalydetection.io.tbwriter import TBWriter
from marineanomalydetection.trainmode import TrainMode
from marineanomalydetection.parse_args_train import parse_args_train
from marineanomalydetection.io.model_handler import (
    load_model,
    save_model,
    get_model_name,
)
from marineanomalydetection.utils.seed import set_seed, set_seed_worker
from marineanomalydetection.dataset.update_class_distribution import (
    update_class_distribution,
)
from marineanomalydetection.utils.device import get_device, empty_cache
from marineanomalydetection.utils.train_functions import (
    train_step_supervised,
    train_step_semi_supervised_separate_batches,
    train_step_semi_supervised_one_batch,
    eval_step,
    get_criterion,
    get_optimizer,
    get_lr_scheduler,
    get_model,
    check_num_alphas,
    get_output_channels,
    get_lr_steps,
    update_checkpoint_path,
    check_checkpoint_path_exist,
)
from marineanomalydetection.dataset.augmentation.get_transform_train import (
    get_transform_train,
)
from marineanomalydetection.dataset.augmentation.get_transform_val import (
    get_transform_val,
)
from marineanomalydetection.io.wandb_logger import WandbLogger
from marineanomalydetection.dataset.augmentation.weakaugmentation import (
    WeakAugmentation,
)
from marineanomalydetection.io.log_functions import log_epoch_init


def main(options, wandb_logger):
    # Use gpu or cpu
    device = get_device()
    file_io = FileIO()
    # Reproducibility
    seed = options["seed"]
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    

    logger.add(
        os.path.join(f"{options['log_folder']}", LOG_SET + ".log"), 
        filter=lambda record: record["extra"].get("name") == LOG_SET
    )
    logger.add(
        sys.__stdout__, 
        filter=lambda record: record["extra"].get("name") == LOG_STD_OUT
    )
    logger.add(
        os.path.join(f"{options['log_folder']}", LOG_SSL_LOSS + ".log"), 
        filter=lambda record: record["extra"].get("name") == LOG_SSL_LOSS
    )
    logger_std_out = logger.bind(name=LOG_STD_OUT)
    logger_ssl_loss = logger.bind(name=LOG_SSL_LOSS)

    model_name = get_model_name(
        options["resume_model"],
        options["mode"],
        options["aggregate_classes"],
        options["today_str"],
        options["run_id"],
        options["run_name"],
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
    transform_val = get_transform_val()
    if options["mode"] == TrainMode.TRAIN_SSL_TWO_TRAIN_SETS \
        or options["mode"] == TrainMode.TRAIN_SSL_ONE_TRAIN_SET:
        weakly_transform = WeakAugmentation(mean=None, std=None)
    # TODO: modify class_distr when using ssl
    # (because you take a percentage of labels so the class distr of pixels
    # will change)
    class_distr = None  # CLASS_DISTR
    standardization = None  # transforms.Normalize(BANDS_MEAN, BANDS_STD)
    # Construct Data loader
    if options["mode"] == TrainMode.TRAIN:
        train_loader, val_loader = get_dataloaders_supervised(
            splits_path=options["splits_path"],
            patches_path=options["patches_path"],
            transform_train=transform_train,
            transform_val=transform_val,
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
    elif options["mode"] == TrainMode.TRAIN_SSL_TWO_TRAIN_SETS:
        (
            labeled_train_loader,
            unlabeled_train_loader,
            val_loader,
        ) = get_dataloaders_ssl_separate_train_sets(
            splits_path=options["splits_path"],
            patches_path=options["patches_path"],
            transform_train=transform_train,
            transform_val=transform_val,
            weakly_transform=weakly_transform,
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
    elif options["mode"] == TrainMode.TRAIN_SSL_ONE_TRAIN_SET:
        train_loader, val_loader = get_dataloaders_ssl_single_train_set(
            splits_path=options["splits_path"],
            patches_path=options["patches_path"],
            transform_train=transform_train,
            transform_val=transform_val,
            weakly_transform=weakly_transform,  
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
    else:
        raise Exception("The specified mode option does not exist.")

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
    if options["mode"] == TrainMode.TRAIN_SSL_TWO_TRAIN_SETS \
        or options["mode"] == TrainMode.TRAIN_SSL_ONE_TRAIN_SET:
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
        val_losses_avg_all_epochs = []
        min_val_loss_among_epochs = float("inf")
        
        dataiter = iter(train_loader)
        image_temp, _ = next(dataiter)
        # Write model-graph to Tensorboard
        tb_writer.add_graph(model, image_temp.to(device))

        ###############################################################
        # Start Supervised Training                                   #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            log_epoch_init(epoch, logger_std_out)
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

                wandb_logger.log_train_loss(
                    loss, 
                    epoch, 
                    len(train_loader), 
                    i_board
                )
                i_board += 1

            logging.info(
                "Training loss was: "
                + str(sum(training_loss) / training_batches)
            )

            # Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                val_losses = []
                val_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(val_loader, desc="validation"):
                        y_predicted, y_true = eval_step(
                            image,
                            target,
                            criterion,
                            val_losses,
                            y_predicted,
                            y_true,
                            model,
                            output_channels,
                            device,
                        )
                        val_batches += target.shape[0]
                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    # Save Scores to the .log file and visualize also with tensorboard
                    acc = Evaluation(y_predicted, y_true)
                    
                    train_loss = sum(training_loss) / training_batches
                    val_loss = sum(val_losses) / val_batches
                    val_losses_avg_all_epochs.append(val_loss)
                    if min(val_losses_avg_all_epochs) < min_val_loss_among_epochs:
                        min_val_loss_among_epochs = min(val_losses_avg_all_epochs)
                        epoch_min_val_loss = epoch
                    logging.info("\n")
                    logging.info(
                        "Val loss was: " + str(val_loss)
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
                            "Val loss": val_loss,
                            "Train loss": train_loss,
                        },
                        epoch,
                    )
                    tb_writer.add_eval_metrics(acc, epoch)
                    
                    wandb_logger.log_eval_losses(
                        train_loss, 
                        val_loss, 
                        min_val_loss_among_epochs, 
                        epoch,
                        epoch_min_val_loss
                    )
                    
                #val_loss = sum(val_losses) / val_batches
                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                model.train()

    elif options["mode"] == TrainMode.TRAIN_SSL_TWO_TRAIN_SETS:
        val_losses_avg_all_epochs = []
        min_val_loss_among_epochs = float("inf")
        classes_channel_idx = 1

        labeled_iter = iter(labeled_train_loader)
        unlabeled_iter = iter(unlabeled_train_loader)
        ###############################################################
        # Start SEMI-SUPERVISED LEARNING Training                     #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            log_epoch_init(epoch, logger_std_out)

            training_loss = []
            training_batches = 0

            i_board = 0
            for _ in tqdm(range(len(labeled_iter)), desc="training"):
                loss, training_loss = train_step_semi_supervised_separate_batches(
                    labeled_train_loader,
                    unlabeled_train_loader,
                    labeled_iter,
                    unlabeled_iter,
                    criterion,
                    criterion_unsup,
                    training_loss,
                    model,
                    optimizer,
                    device,
                    options["batch"],
                    classes_channel_idx,
                    options["threshold"],
                    options["lambda_coeff"],
                    logger_ssl_loss,
                    PADDING_VAL,
                )
                training_batches += options["batch"]
                # Write running loss
                tb_writer.add_scalar(
                    "training loss",
                    loss,
                    (epoch - 1) * len(labeled_train_loader) + i_board,
                )
                wandb_logger.log_train_loss(
                    loss, 
                    epoch, 
                    len(labeled_train_loader), 
                    i_board
                )
                i_board += 1

            logging.info("Training loss was: " + str(np.mean(training_loss)))

            # Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                val_losses = []
                val_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(val_loader, desc="validation"):
                        y_predicted, y_true = eval_step(
                            image,
                            target,
                            criterion,
                            val_losses,
                            y_predicted,
                            y_true,
                            model,
                            output_channels,
                            device,
                        )
                        val_batches += target.shape[0]

                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    # Save Scores to the .log file and visualize also with tensorboard
                    acc = Evaluation(y_predicted, y_true)
                    val_loss = sum(val_losses) / val_batches
                    val_losses_avg_all_epochs.append(val_loss)
                    if min(val_losses_avg_all_epochs) < min_val_loss_among_epochs:
                        min_val_loss_among_epochs = min(val_losses_avg_all_epochs)
                        epoch_min_val_loss = epoch
                    
                    logging.info("\n")
                    logging.info(
                        "Val loss was: " + str(val_loss)
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
                            "Val loss": val_loss,
                            "Train loss": np.mean(training_loss),
                        },
                        epoch,
                    )
                    tb_writer.add_eval_metrics(acc, epoch)
                    
                    wandb_logger.log_eval_losses(
                        np.mean(training_loss), 
                        val_loss, 
                        min_val_loss_among_epochs, 
                        epoch,
                        epoch_min_val_loss
                    )

                val_loss = sum(val_losses) / val_batches
                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                model.train()
    elif options["mode"] == TrainMode.TRAIN_SSL_ONE_TRAIN_SET:
        val_losses_avg_all_epochs = []
        min_val_loss_among_epochs = float("inf")
        classes_channel_idx = 1
        
        #dataiter = iter(train_loader)
        #image_temp, _ = next(dataiter)
        # Write model-graph to Tensorboard
        #tb_writer.add_graph(model, image_temp.to(device))

        ###############################################################
        # Start Supervised Training                                   #
        ###############################################################
        model.train()

        for epoch in range(start, epochs + 1):
            log_epoch_init(epoch, logger_std_out)
            training_loss = []
            training_batches = 0

            i_board = 0
            for image, target, weakly_aug_image in tqdm(train_loader, desc="training"):
                loss, training_loss = train_step_semi_supervised_one_batch(
                    image=image,
                    seg_map=target,
                    weak_aug_img=weakly_aug_image,
                    criterion=criterion,
                    criterion_unsup=criterion_unsup,
                    training_loss=training_loss,
                    model=model,
                    optimizer=optimizer,
                    device=device,
                    batch_size=options["batch"],
                    classes_channel_idx=classes_channel_idx,
                    threshold=options["threshold"],
                    lambda_v=options["lambda_coeff"],
                    logger_ssl_loss=logger_ssl_loss,
                    padding_val=PADDING_VAL,
                )
                training_batches += target.shape[0]
                # Write running loss
                tb_writer.add_scalar(
                    "training loss",
                    loss,
                    (epoch - 1) * len(train_loader) + i_board,
                )

                wandb_logger.log_train_loss(
                    loss, 
                    epoch, 
                    len(train_loader), 
                    i_board
                )
                i_board += 1

            logging.info(
                "Training loss was: "
                + str(sum(training_loss) / training_batches)
            )

            # Evaluation
            if epoch % eval_every == 0 or epoch == 1:
                model.eval()

                val_losses = []
                val_batches = 0
                y_true = []
                y_predicted = []

                with torch.no_grad():
                    for image, target in tqdm(val_loader, desc="validation"):
                        y_predicted, y_true = eval_step(
                            image,
                            target,
                            criterion,
                            val_losses,
                            y_predicted,
                            y_true,
                            model,
                            output_channels,
                            device,
                        )
                        val_batches += target.shape[0]
                    y_predicted = np.asarray(y_predicted)
                    y_true = np.asarray(y_true)
                    # Save Scores to the .log file and visualize also with tensorboard
                    acc = Evaluation(y_predicted, y_true)
                    
                    train_loss = sum(training_loss) / training_batches
                    val_loss = sum(val_losses) / val_batches
                    val_losses_avg_all_epochs.append(val_loss)
                    if min(val_losses_avg_all_epochs) < min_val_loss_among_epochs:
                        min_val_loss_among_epochs = min(val_losses_avg_all_epochs)
                        epoch_min_val_loss = epoch
                    logging.info("\n")
                    logging.info(
                        "Val loss was: " + str(val_loss)
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
                            "Val loss": val_loss,
                            "Train loss": train_loss,
                        },
                        epoch,
                    )
                    tb_writer.add_eval_metrics(acc, epoch)
                    
                    wandb_logger.log_eval_losses(
                        train_loss, 
                        val_loss, 
                        min_val_loss_among_epochs, 
                        epoch,
                        epoch_min_val_loss
                    )
                    
                #val_loss = sum(val_losses) / val_batches #TODO ?
                if options["reduce_lr_on_plateau"] == 1:
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

                model.train()


if __name__ == "__main__":
    wandb_logger = WandbLogger()
    wandb_logger.login()
    config = wandb_logger.get_config("./config.yaml")

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
    main(options, wandb_logger)
