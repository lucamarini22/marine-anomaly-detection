<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#set-up-dataset">Set-up datasets</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a>
    <ul>
        <li><a href="#training">Training</a></li>
        <li><a href="#evaluation">Evaluation</a></li>
      </ul>
    </li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About the project
The marine-anomaly-detection project provides code to apply the state of the art of semi-supervised learning techniques to marine anomaly detection semantic segmentation problems on satellite imagery of marine regions. The considered anomalies are marine-litter (marine debris), ships, clouds, and algae/organic materials.

The code builds on and extends the following two repositories:
- [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). Compared to the original repository, this repository adapts FixMatch to be used for semantic segmentation tasks and to work with multispectral images.
- [marine-debris.github.io](https://github.com/marine-debris/marine-debris.github.io), which provides the code to work with the [MARIDA](https://marine-debris.github.io/index.html) dataset.

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

It is recommended to use [conda](https://docs.conda.io/en/latest/) to set-up the environment. [conda](https://docs.conda.io/en/latest/) will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/lucamarini22/marine-anomaly-detection/blob/main/environment.yml).

### Installation

1. Get [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba), or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), or similar. Micromamba is preferred to Miniconda for its greater speed in creating the virtual environment.
2. Clone the repo.
   ```sh
   git clone https://github.com/lucamarini22/marine-anomaly-detection.git
   ```
3. Setup and activate the environment. This will create a conda environment called `marine-anomaly-detection`.
   ```sh
   micromamba env create -f environment.yml
   ```
   ```sh
   micromamba activate marine-anomaly-detection
   ```
4. [Optional] Install the local package.
   ```
   pip install -e .
   ```

### Set up dataset
To launch the training on MARIDA, it is necessary to download the dataset. The dataset can be downloaded [here](https://zenodo.org/record/5151941) and has the following structure:
   
   - `patches`: folder containing the patches (multispectral images).
   - `splits`: folder containing split files of the training, validation, and test sets.

The `--patches_path` and `--splits_path` arguments in `marineanomalydetection/parse_args_train.py` file shall be respectively point to the `patches` and `splits` folders.


<!-- USAGE EXAMPLES -->
## Usage

### Training
1. Create a [Weight and Biases](https://wandb.ai) account to keep track of the experiments.
2. Set the values of the hyperparameters in this [config.yaml](https://github.com/lucamarini22/marine-anomaly-detection/blob/main/config.yaml) file.
3. Enter in the main folder
   ```sh
   cd /marine-anomaly-detection/
   ```
4. Create a [Sweep](https://docs.wandb.ai/guides/sweeps) to keep track of your training runs.
   ```
   wandb sweep --project <project-name> config.yaml
   ```
5. Start an agent and execute $NUM training runs.
   ```
   wandb agent --count $NUM <your-entity/sweep-demo-cli/sweepID>
   ```

### Evaluation
1. Evaluate a model.
   ```
   python evaluation.py --model_path=<path_to_model>
   ```
2. Visualize the predictions of the last evaluated model by running the cells of the notebook [Visualize Predictions.ipynb](https://github.com/lucamarini22/marine-anomaly-detection/blob/main/notebooks/Visualize%20Predictions.ipynb). Specify the variable `tile_name` to see the predictions for the patches of the specified tile.



<!-- ACKNOWLEDGEMENTS 
This README was based on https://github.com/othneildrew/Best-README-Template
-->