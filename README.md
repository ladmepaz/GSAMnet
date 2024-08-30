# DiagAssistAI
![GitHub License](https://img.shields.io/github/license/WilhelmBuitrago/DiagAssistAI)
![Static Badge](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)
<p float="center">
  <img src=".asset/model_p.png?raw=true"/>
</p>

# Installation

## Conda Installation

To install Conda locally, follow the instructions provided in the [official Conda documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

### Creating a Conda Environment

Create a Conda environment with Python 3.10.12 using the following command:

    conda create -n <name> python=3.10.12


Activate the environment with:

    conda activate <name>

### Installing Packages

#### Installing torch

It's necessary to install PyTorch first following instructions at [https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally). In this project, **CUDA 12.1** was used. You can install PyTorch with support for CUDA 12.1 using the following command: 

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121


## PyPI Package Installation

### Installing GSamNetwork

GSamNetwork requires `python==3.10.12`, as well as `torch==2.4.0`.

To install GSamNetwork, use the following command:

    pip install groundino-samnet
