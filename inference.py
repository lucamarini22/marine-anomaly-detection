import os
import logging
import time
import rasterio
import argparse
import numpy as np
from tqdm import tqdm
from os.path import dirname

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from marineanomalydetection.models.unet import UNet
from marineanomalydetection.dataset.mad_labeled import (
    MADLabeled
)
from marineanomalydetection.utils.constants import BANDS_MEAN, BANDS_STD
from marineanomalydetection.io.load_roi import load_roi
from marineanomalydetection.utils.metrics import Evaluation, confusion_matrix
from marineanomalydetection.utils.assets import (
    labels_binary,
    labels_multi,
)
from marineanomalydetection.dataset.categoryaggregation import (
    CategoryAggregation,
)
from marineanomalydetection.dataset.dataloadertype import DataLoaderType
from marineanomalydetection.utils.seed import set_seed
from marineanomalydetection.utils.train_functions import get_model
from marineanomalydetection.utils.set_bool_flag import set_bool_flag


root_path = dirname(os.path.abspath(__file__))


def inference(options):
    set_seed(options["seed"])

    # Transformations
    transform_test = transforms.Compose([transforms.ToTensor()])

    # Aggregate Distribution Mixed Water, Wakes, Cloud Shadows, Waves with Marine Water
    if options["aggregate_classes"] == CategoryAggregation.MULTI:
        # Keep Marine Debris, Algae/Natural Organic Material, Ship, Clouds, Marine Water classes
        labels = labels_multi
        output_channels = len(labels_multi)
    elif options["aggregate_classes"] == CategoryAggregation.BINARY:
        # Keep only Marine Debris and Others classes
        labels = labels_binary
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

    model = get_model(
        input_bands=options["input_channels"],
        output_classes=output_channels,
        hidden_channels=options["hidden_channels"],
    )

    model.to(device)

    # Load model from specific epoch to continue the training or start the evaluation
    model_file = options["model_path"]
    logging.info("Loading model files from folder: %s" % model_file)

    checkpoint = torch.load(model_file, map_location=device)
    model.load_state_dict(checkpoint)

    del checkpoint  # dereference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    model.eval()

    image = ...
    image = image.to(device)

    with torch.no_grad():
        start = time.time()

        logits = model(image)

        end = time.time()
        print(end - start)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Options
    parser.add_argument(
        "--seed",
        default=0,
        help=("Seed."),
        type=int,
    )
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

    parser.add_argument(
        "--batch", default=5, type=int, help="Number of epochs to run"
    )

    # Unet parameters
    parser.add_argument(
        "--input_channels", default=11, type=int, help="Number of input bands"
    )

    parser.add_argument(
        "--hidden_channels",
        default=16,
        type=int,
        help="Number of hidden features",
    )
    # Data parameters
    parser.add_argument(
        "--use_l1c",
        type=int,
        help="0 to train on L1C data. 1 to train on MARIDA data (atmospherically corrected data).",
        choices=[0, 1]
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
    # Unet model path
    parser.add_argument(
        "--model_path",
        default=os.path.join(
            "results",
            "trained_models",
            "semi-supervised-one-train-set",
            "2023_06_01_H_11_19_30_TRAIN_SSL_ONE_TRAIN_SET_MULTI_x9cs392u_northern-sweep-8",
            "638",
            "model.pth",
        ),
        help="Path to trained model",
    )

    # Produce Predicted Masks
    parser.add_argument(
        "--predict_masks",
        type=int,
        choices=[0, 1],
        default=1,
        help="Generate test set prediction masks?",
    )
    parser.add_argument(
        "--gen_masks_path",
        default=os.path.join(root_path, "data", "predicted_unet"),
        help="Path to where to produce store predictions",
    )

    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict
        # Converts boolean args from [0, 1] to [False, True]
    bool_args_names = [
        "use_l1c",
        "predict_masks"
    ]
    for bool_arg_name in bool_args_names:
        options[bool_arg_name] = set_bool_flag(options[bool_arg_name])

    inference(options)
