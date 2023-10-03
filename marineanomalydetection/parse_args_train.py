import os
import argparse
import wandb
import random

from marineanomalydetection.utils.string import get_today_str
from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.trainmode import TrainMode


def parse_args_train():
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"] = "1000"
    os.environ["WANDB_AGENT_DISABLE_FLAPPING"] = "true"
    parser = argparse.ArgumentParser()
    today_str = get_today_str()

    # Options
    parser.add_argument(
        "--seed",
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
            eleven (Marine Debris, Dense Sargassum, Sparse Sargassum, \
                Natural Organic Material, Ship, Clouds, Marine Water, \
                Sediment-Laden Water, Foam, Turbid Water, Shallow Water); \
            None (keep the original 15 classes)",
    )
    # Training mode
    parser.add_argument(
        "--mode",
        help="Mode",
    )
    # SSL hyperparameters
    parser.add_argument(
        "--perc_labeled",
        help=(
            "Percentage of labeled training set. This argument has "
            "effect only when --mode=TrainMode.TRAIN_SSL_TWO_TRAIN_SETS. "
            " The percentage of the unlabeled training set will be "
            " 1 - perc_labeled."
        ),
        type=float,
    )
    parser.add_argument(
        "--mu",
        help=("Unlabeled data ratio."),
        type=float,
    )
    parser.add_argument(
        "--threshold",
        help=("Confidence (probability) threshold for pseudo-labels."),
        type=float,
    )
    parser.add_argument(
        "--lambda_coeff",
        type=float,
        help="Coefficient of unlabeled loss.",
    )
    # Focal loss
    parser.add_argument(
        "--gamma",
        default=2.0,
        type=float,
        help=("Gamma coefficient of the focal loss of only the supervised "
              "component of the loss. The unsupervised component of the loss "
              "has gamma = 0, which correspond to computing the weighted " 
              "cross-entropy."),
    )
    parser.add_argument(
        "--alphas",
        default=[1.0, 1.0, 1.0, 1.0, 1.0], # [0.50, 0.125, 0.125, 0.125, 0.125], #
        type=list[float],
        help="Alpha coefficients of the focal loss.",
    )
    parser.add_argument(
        "--use_ce_in_unsup_component",
        default=True,
        type=bool,
        help=("Only to consider when training with semi-supervised learning. "
        "True to use the Cross Entropy loss in the unsupervised component "
        "of the loss. False to use the Focal loss. The Focal loss is not "
        "ideal for the unsupervised component when used with a high "
        "probability threshold (--threshold) because the Focal loss gives "
        "more importance to predictions with a low probability, which are "
        "ignored when having a high confidence (probability) threshold."
        ),
    )
    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs to run",
    )
    parser.add_argument("--batch", type=int, help="Batch size")
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
        "--patches_path", 
        help="path of the folder containing the patches", 
        default=os.path.join("data", "patches")
    )
    parser.add_argument(
        "--splits_path",
        help="path of the folder containing the splits files", 
        default=os.path.join("data", "splits")
    )
    # Optimization
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument("--decay", default=0, type=float, help="Weight decay")
    parser.add_argument(
        "--reduce_lr_on_plateau",
        type=int,
        help="Reduces learning rate when no increase (0 or 1).",
    )
    parser.add_argument(
        "--lr_steps",
        default="[10000]",
        type=str,
        help="Specify the steps that the lr will be reduced. To use only when reduce_lr_on_plateau is set to 0 in the config.yaml file. When reduce_lr_on_plateau = 1 another learning rate decay strategy is applied.",
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
    options["run_id"] = wandb.run.id
    options["run_name"] = wandb.run.name
    
    
    """For Debugging
    options["mode"] = "TrainMode.TRAIN_SSL_TWO_TRAIN_SETS" #TRAIN_SSL_ONE_TRAIN_SET" #TRAIN_SSL_TWO_TRAIN_SETS" # TRAIN_SSL_ONE_TRAIN_SET" #
    options["lr"] = 2e-4
    options["threshold"] = 0.0
    options["epochs"] = 2000
    options["batch"] = 5
    options["seed"] = random.randint(0, 1000)
    options["reduce_lr_on_plateau"] = 0
    options["lambda_coeff"] = 1.0
    options["mu"] = 5
    options["perc_labeled"] = 0.1
    """

    options["mode"] = TrainMode[str(options["mode"]).split(".")[-1]]
    
    if options["mode"] == TrainMode.TRAIN_SSL_TWO_TRAIN_SETS:  
        options["mu"] = int(options["mu"])
        if options["perc_labeled"] <= 0.0 or options["perc_labeled"] >= 1.0:
            raise Exception("The parameter 'perc_labeled' should have a value in the interval ]0.0, 1.0[")

    if options["perc_labeled"] == 0.9:
        options["seed"] = 498
    elif options["perc_labeled"] == 0.8:
        options["seed"] = 931
    elif options["perc_labeled"] == 0.7:
        options["seed"] = 212
    elif options["perc_labeled"] == 0.6:
        options["seed"] = 450
    elif options["perc_labeled"] == 0.5:
        options["seed"] = 148
    elif options["perc_labeled"] == 0.4:
        options["seed"] = 243
    elif options["perc_labeled"] == 0.3:
        options["seed"] = 332
    elif options["perc_labeled"] == 0.2:
        options["seed"] = 318
    elif options["perc_labeled"] == 0.1:
        options["seed"] = 949
    elif options["perc_labeled"] == 0.05:
        options["seed"] = 640

    #with open("/data/anomaly-marine-detection/data/splits/seed.txt", "a") as myfile:
    #    myfile.write(str(options["seed"]) + "\n")
    #print(options["seed"])    
    
    options["batch"] = int(options["batch"])

    return options
