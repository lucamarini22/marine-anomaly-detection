## About the project
The anomaly-marine-detection project provides code to apply the state of the art of semi-supervised learning techniques to anomaly detection semantic segmentation problems on satellite imagery of marine regions. The considered anomalies are marine-litter (marine debris), ships, clouds, and algae/organic materials.

The code builds on and extends the following two repositories:
- [FixMatch-pytorch](https://github.com/kekmodel/FixMatch-pytorch) implementation based on [PyTorch](https://pytorch.org/). Compared to the original repository, this repository adapts FixMatch to be used for semantic segmentation tasks and to work with multispectral images.
- [marine-debris.github.io](https://github.com/marine-debris/marine-debris.github.io), which provides the code to work with the [MARIDA](https://marine-debris.github.io/index.html) dataset.

## Getting Started

### Prerequisites

It is recommended to use [conda](https://docs.conda.io/en/latest/) to set-up the environment. [conda](https://docs.conda.io/en/latest/) will take care of all requirements for you. For a detailed list of required packages, please refer to the [conda environment file](https://github.com/lucamarini22/anomaly-marine-detection/blob/main/environment.yml).

### Installation

1. Get [micromamba](https://mamba.readthedocs.io/en/latest/installation.html#micromamba) or similar.
2. Clone the repo.
   ```sh
   git clone https://github.com/lucamarini22/anomaly-marine-detection.git
   ```
3. Setup and activate the environment. This will create a conda environment called `anomaly-marine-detection`.
   ```sh
   micromamba env create -f environment.yml
   ```
   ```sh
   micromamba activate anomaly-marine-detection
   ```
4. Install the local package.
   ```
   pip install -e .
   ```

