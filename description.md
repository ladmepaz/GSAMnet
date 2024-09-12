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

## Version 0.4.4
### Added:

Added: FPS calculation can now be performed with either a list of images or a DataLoader.

### Fixed:

Fixed: Error in batch mode calculation in SAM2.

Fixed: Changed the way the batch is calculated for SAM2 (removed the default batch implementation of SAM, it does not support large volumes of images).



