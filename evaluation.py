import os
import logging
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

root_path = dirname(os.path.abspath(__file__))


def main(options):
    
    if not os.path.isdir(options["log_folder"]):
        raise Exception(f"The log folder '{options['log_folder']}' does not exist. Please create it.")
    
    logging.basicConfig(
        filename=os.path.join(
            root_path, 
            options["log_folder"], 
            options["log_file"]
        ),
        filemode="a",
        level=logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    logging.info("*" * 10)
    
    set_seed(options["seed"])
    # Transformations

    transform_test = transforms.Compose([transforms.ToTensor()])
    standardization = None  # transforms.Normalize(BANDS_MEAN, BANDS_STD)

    # Construct Data loader

    dataset_test = MADLabeled(
        DataLoaderType.TEST_SET,
        transform=transform_test,
        standardization=standardization,
        aggregate_classes=options["aggregate_classes"],
        patches_path=options["patches_path"],
        splits_path=options["splits_path"],
    )

    test_loader = DataLoader(
        dataset_test, batch_size=options["batch"], shuffle=False
    )
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

    model = UNet(
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

    y_true = []
    y_predicted = []

    with torch.no_grad():
        for image, target in tqdm(test_loader, desc="testing"):
            image = image.to(device)
            target = target.to(device)

            logits = model(image)

            # Accuracy metrics only on annotated pixels
            logits = torch.movedim(logits, (0, 1, 2, 3), (0, 3, 1, 2))
            logits = logits.reshape((-1, output_channels))
            target = target.reshape(-1)
            mask = target != -1
            logits = logits[mask]
            target = target[mask]

            probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()
            target = target.cpu().numpy()

            y_predicted += probs.argmax(1).tolist()
            y_true += target.tolist()

        ####################################################################
        # Save Scores to the .log file                                     #
        ####################################################################
        acc = Evaluation(y_predicted, y_true)
        logging.info("\n")
        logging.info("STATISTICS: \n")
        logging.info("Evaluation: " + str(acc))
        print("Evaluation: " + str(acc))
        conf_mat = confusion_matrix(y_true, y_predicted, labels)
        logging.info("Confusion Matrix:  \n" + str(conf_mat.to_string()))
        print("Confusion Matrix:  \n" + str(conf_mat.to_string()))

        if options["predict_masks"]:
            path = os.path.join(root_path, "data", "patches")
            ROIs = load_roi(
                os.path.join(root_path, "data", "splits", "test_X.txt")
            )

            impute_nan = np.tile(BANDS_MEAN, (256, 256, 1))

            for roi in tqdm(ROIs):
                roi_folder = "_".join(
                    ["S2"] + roi.split("_")[:-1]
                )  # Get Folder Name
                roi_name = "_".join(["S2"] + roi.split("_"))  # Get File Name
                roi_file = os.path.join(
                    path, roi_folder, roi_name + ".tif"
                )  # Get File path

                os.makedirs(options["gen_masks_path"], exist_ok=True)

                output_image = os.path.join(
                    options["gen_masks_path"],
                    os.path.basename(roi_file).split(".tif")[0] + "_unet.tif",
                )

                # Read metadata of the initial image
                with rasterio.open(roi_file, mode="r") as src:
                    tags = src.tags().copy()
                    meta = src.meta
                    image = src.read()
                    image = np.moveaxis(image, (0, 1, 2), (2, 0, 1))
                    dtype = src.read(1).dtype

                # Update meta to reflect the number of layers
                meta.update(count=1)
                if os.path.isfile(output_image):
                    os.remove(output_image)
                # Write it
                with rasterio.open(output_image, "w", **meta) as dst:
                    # Preprocessing before prediction
                    nan_mask = np.isnan(image)
                    image[nan_mask] = impute_nan[nan_mask]

                    image = transform_test(image)
                    if standardization is not None:
                        image = standardization(image)

                    # Image to Cuda if exist
                    image = image.to(device)

                    # Predictions
                    logits = model(image.unsqueeze(0))

                    probs = (
                        torch.nn.functional.softmax(logits.detach(), dim=1)
                        .cpu()
                        .numpy()
                    )

                    probs = probs.argmax(1).squeeze() + 1

                    # Write the mask with georeference
                    dst.write_band(
                        1, probs.astype(dtype).copy()
                    )  # In order to be in the same dtype
                    dst.update_tags(**tags)


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
            "2023_07_05_H_08_50_44_TRAIN_SSL_ONE_TRAIN_SET_MULTI_jisg3mrz_drawn-sweep-1",
            "1133",
            "model.pth",
        ),
        help="Path to trained model",
    )

    # Produce Predicted Masks
    parser.add_argument(
        "--predict_masks",
        default=True,
        type=bool,
        help="Generate test set prediction masks?",
    )
    parser.add_argument(
        "--gen_masks_path",
        default=os.path.join(root_path, "data", "predicted_unet"),
        help="Path to where to produce store predictions",
    )
    
    parser.add_argument(
        "--log_folder",
        default="logs",
        type=str,
        help="Path of the log folder",
    )
    parser.add_argument(
        "--log_file",
        default="evaluating_unet.log",
        type=str,
        help="Name of log file.",
    )
    
    args = parser.parse_args()
    options = vars(args)  # convert to ordinary dict

    main(options)
