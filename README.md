
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
4. Install the local package.
   ```
   pip install -e .
   ```

