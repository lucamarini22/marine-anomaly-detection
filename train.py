import os
import ast
import json
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms

from anomalymarinedetection.utils.assets import labels_binary, labels_multi
from anomalymarinedetection.loss.focal_loss import FocalLoss
from anomalymarinedetection.models.unet import UNet
from anomalymarinedetection.dataset.get_dataloaders import (
    get_dataloaders_supervised,
    get_dataloaders_ssl,
    get_dataloaders_eval,
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
    IGNORE_INDEX,
    PADDING_VAL,
)
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.dataset.dataloadertype import DataLoaderType
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
from anomalymarinedetection.utils.check_num_alphas import check_num_alphas
from anomalymarinedetection.dataset.update_class_distribution import (
    update_class_distribution,
)


def main(options):
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
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            DiscreteRandomRotation([-90, 0, 90, 180]),
            transforms.RandomHorizontalFlip(),
        ]
    )
    transform_test = transforms.Compose([transforms.ToTensor()])
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
    elif options["mode"] == TrainMode.EVAL:
        test_loader = get_dataloaders_eval(
            dataset_path=options["dataset_path"],
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
    else:
        raise Exception("The mode option should be train, train_ssl, or test")

    if options["aggregate_classes"] == CategoryAggregation.MULTI:
        output_channels = len(labels_multi)
    elif options["aggregate_classes"] == CategoryAggregation.BINARY:
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        start = int(options["resume_model"].split("/")[-2]) + 1
    else:
        start = 1

    if class_distr is not None:
        class_distr = update_class_distribution(
            options["aggregate_classes"], class_distr
        )
    # Coefficients of Focal loss
    alphas = torch.Tensor([0.50, 0.125, 0.125, 0.125, 0.125])  # 1 / class_distr
    check_num_alphas(alphas, output_channels)
    # Init of supervised loss
    criterion = FocalLoss(
        alpha=alphas.to(device),
        gamma=2.0,
        reduction="mean",
        ignore_index=IGNORE_INDEX,
    )
    if options["mode"] == TrainMode.TRAIN_SSL:
        # Init of unsupervised loss
        criterion_unsup = FocalLoss(
            alpha=alphas.to(device),
            gamma=2.0,
            reduction="none",
            ignore_index=IGNORE_INDEX,
        )
    # Optimizer
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

            # Evaluation
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
                strong_transform = StrongAugmentation(randaugment=randaugment, 
                                                      mean=None, std=None)
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
                    # c = logits_u_w_i[:, :, 0]
                    # d = logits_u_w_i[:, :, 1]
                    logits_u_w_i = strong_transform(logits_u_w_i)
                    # e = logits_u_w_i[0, :, :]
                    # f = logits_u_w_i[1, :, :]  # TODO: visually debug these
                    tmp[i, :, :, :] = logits_u_w_i
                    # g = logits_u_s[i, 0, :, :]
                    # h = logits_u_s[i, 1, :, :]
                    # aa = img_x[i, 7, :, :]
                    # bb = seg_map[i, :, :].float()
                    # cc = img_u_w[i, 4, :, :]
                    # dd = img_u_s[i, 4, :, :]
                    # print()

                logits_u_w = tmp
                logits_u_s = logits_u_s.cpu().detach().numpy()

                for i in range(logits_u_w.shape[0]):
                    for j in range(logits_u_w.shape[1]):
                        logits_u_w_i_j = logits_u_w[i, j, :, :]
                        logits_u_s_i_j = logits_u_s[i, j, :, :]
                        # a = logits_u_w_i_j
                        # c = logits_u_s_i_j
                        logits_u_s_i_j[
                            np.where(logits_u_w_i_j == PADDING_VAL)
                        ] = IGNORE_INDEX
                        # b = logits_u_s_i_j
                        logits_u_s_i_j = torch.from_numpy(logits_u_s_i_j)
                        logits_u_s[i, j, :, :] = logits_u_s_i_j
                        # z = logits_u_s[i, j, :, :]
                        # print()

                logits_u_w = torch.from_numpy(logits_u_w)
                logits_u_w = logits_u_w.to(device)

                logits_u_s = torch.from_numpy(logits_u_s)
                logits_u_s = logits_u_s.to(device)

                pseudo_label = torch.softmax(
                    logits_u_w.detach(), dim=-1
                )  # / args.T, dim=-1) -> to add temperature
                # target_u contains the idx of the class having the highest
                # probability (for all pixels and for all images in the batch)
                max_probs, targets_u = torch.max(
                    pseudo_label, dim=classes_channel_idx
                )  # dim=-1)
                mask = max_probs.ge(options["threshold"]).float()
                padding_mask = logits_u_s[:, 0, :, :] == IGNORE_INDEX
                mask[padding_mask] = 0
                # targets_u[logits_u_s[:, 0, :, :] == IGNORE_INDEX] = IGNORE_INDEX

                """ # Debug visually
                for i in range(logits_u_s.shape[0]):
                    a = logits_u_s[i, 0, :, :]
                    b = logits_u_s[i, 1, :, :]
                    c = logits_u_s[i, 2, :, :]
                    d = logits_u_s[i, 3, :, :]
                    e = logits_u_s[i, 4, :, :]
                    f = targets_u[i, :, :]
                    plt.imshow(f.cpu().detach().numpy())
                    plt.savefig("results/a.png")
                    f[padding_mask[0, :, :]] = PADDING_VAL
                    print()
                """

                # Unsupervised loss
                loss_u = criterion_unsup(logits_u_s, targets_u) * torch.flatten(
                    mask
                )
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

            # Evaluation
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
    elif options["mode"] == TrainMode.EVAL:
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
            # Save Scores to the .log file
            acc = Evaluation(y_predicted, y_true)
            logging.info("\n")
            logging.info("Test loss was: " + str(sum(test_loss) / test_batches))
            logging.info("STATISTICS: \n")
            logging.info("Evaluation: " + str(acc))


if __name__ == "__main__":
    options = parse_args_train()

    if options["mode"] == TrainMode.TRAIN_SSL:
        options["checkpoint_path"] = os.path.join(
            options["checkpoint_path"], "semi-supervised"
        )
    elif options["mode"] == TrainMode.TRAIN:
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
