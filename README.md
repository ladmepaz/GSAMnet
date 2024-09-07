# DiagAssistAI

# Model overview
![GitHub License](https://img.shields.io/github/license/WilhelmBuitrago/DiagAssistAI)
![Static Badge](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)
![PyPI - Version](https://img.shields.io/pypi/v/groundino-samnet?pypiBaseUrl=https%3A%2F%2Fpypi.org&link=https%3A%2F%2Fpypi.org%2Fproject%2Fgroundino-samnet%2F)

<p float="center">
  <img src=".asset/model_p.png?raw=true"/>
</p>

This project integrates the **GroundingDINO** model with the **SAM1** and **SAM2** models to achieve accurate detection and segmentation of feet in women in labor.

- **GroundingDINO**: [Repository Link](https://github.com/IDEA-Research/GroundingDINO)
- **SAM1**: [Repository Link](https://github.com/facebookresearch/segment-anything)
- **SAM2**: [Repository Link](https://github.com/facebookresearch/segment-anything-2)

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
