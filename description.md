# DiagAssistAI
![GitHub License](https://img.shields.io/github/license/WilhelmBuitrago/DiagAssistAI)
![Static Badge](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)
![PyPI - Version](https://img.shields.io/pypi/v/groundino-samnet?link=https%3A%2F%2Fpypi.org%2Fproject%2Fgroundino-samnet%2F)

![modelo](https://raw.githubusercontent.com/WilhelmBuitrago/DiagAssistAI/main/.asset/model_p.png)

# Installation

GSamNetwork requires `python==3.10.12`, as well as `torch==2.4.0`.

## Installing PyTorch

Make sure to check the version of Python that is compatible with your CUDA version at the following link: [Installing torch locally](https://pytorch.org/get-started/locally/).

In this project, **CUDA 12.1** was used. You can install PyTorch with support for CUDA 12.1 using the following command:

``
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
``

## Installing Package

To install the package, use:

``
pip install groundino-samnet
``

## Version 0.2.1

### Added

Added: Detailed function descriptions

### Deteled

Deleted: `box_xyxy_to_xywh` function for utils

### Notice

Notice: Notice Regarding the Use of the `cleanup()` Function for Datasets




