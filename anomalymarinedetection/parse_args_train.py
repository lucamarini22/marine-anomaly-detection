import os
import argparse

from anomalymarinedetection.utils.string import get_today_str
from anomalymarinedetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from anomalymarinedetection.trainmode import TrainMode


def parse_args_train(config):
    parser = argparse.ArgumentParser()
    today_str = get_today_str()

    # Options
    parser.add_argument(
        "--seed",
        default=2,
        help=("Seed."),
        type=int,
    )
    # Aggregation of categories
    parser.add_argument(
        "--aggregate_classes",
        choices=list(CategoryAggregation),
        default=CategoryAggregation.MULTI,
        type=str,
        help="Aggregate classes into:\
            multi (Marine Water, Algae/OrganicMaterial, Marine Debris, Ship, and Cloud);\
                binary (Marine Debris and Other); \
                    no (keep the original 15 classes)",
    )
    # Training mode
    parser.add_argument(
        "--mode",
        choices=list(TrainMode),
        default=TrainMode.TRAIN_SSL,
        help="Mode",
    )
    # SSL hyperparameters
    parser.add_argument(
        "--perc_labeled",
        default=0.8,
        help=(
            "Percentage of labeled training set. This argument has "
            "effect only when --mode=TrainMode.TRAIN_SSL. "
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
    # Focal loss
    parser.add_argument(
        "--gamma",
        default=2.0,
        type=float,
        help="Gamma coefficient of the focal loss.",
    )
    parser.add_argument(
        "--alphas",
        default=[0.50, 0.125, 0.125, 0.125, 0.125],
        type=list[float],
        help="Alpha coefficients of the focal loss.",
    )
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        default=20000,
        type=int,
        help="Number of epochs to run",
    )
    parser.add_argument("--batch", default=5, type=int, help="Batch size")
    parser.add_argument(
        "--resume_model",
        default=None,  # "/data/anomaly-marine-detection/results/trained_models/semi-supervised/2023_04_18_H_09_27_31_SSL_multi/1592/model.pth",
        type=str,
        help="Load model from previous epoch. Specify path to the checkpoint.",
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
    # Optimization
    parser.add_argument("--lr", default=2e-4, type=float, help="learning rate")
    parser.add_argument("--decay", default=0, type=float, help="Weight decay")
    parser.add_argument(
        "--reduce_lr_on_plateau",
        default=0,
        type=int,
        help="Reduce learning rate when no increase (0 or 1)",
    )
    parser.add_argument(
        "--lr_steps",
        default="[10000]",
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
    options["lr"] = config.lr
    options["threshold"] = config.threshold
    return options
