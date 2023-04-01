## About the project
The anomaly-marine-detection project provides code to apply the state of the art of semi-supervised learning techniques to anomaly detection semantic segmentation problems on satellite imagery of marine regions.

The code builds on and extends the following two repositories:
- [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). Compared to the original repository, this repository adapts FixMatch to be used for semantic segmentation tasks and to work with multispectral images.
- [marine-debris.github.io](https://github.com/marine-debris/marine-debris.github.io), which provides the code to work with the [MARIDA](https://marine-debris.github.io/index.html) dataset.

## Getting Started

### Prerequisites

It is recommended to use [conda](https://docs.conda.io/en/latest/) to set-up the environment. [conda](https://docs.conda.io/en/latest/) will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/lucamarini22/anomaly-marine-detection/blob/main/environment.yml).

### Installation

1. Get [miniconda](https://docs.conda.io/en/latest/miniconda.html) or similar.
2. Clone the repo.
   ```sh
   git clone https://github.com/lucamarini22/anomaly-marine-detection.git
   ```
3. Setup and activate the environment. This will create a conda environment called `anomaly-marine-detection`.
   ```sh
   conda env create -f environment.yml
   ```
   ```sh
   conda activate anomaly-marine-detection
   ```

