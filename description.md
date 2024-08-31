# DiagAssistAI
![GitHub License](https://img.shields.io/github/license/WilhelmBuitrago/DiagAssistAI)
![Static Badge](https://img.shields.io/badge/Python-3.10.12-blue?logo=python)

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

## Version 0.1.22

### Fixed

Resolved: Error to import utils in sam2 image predictor

Resolved: Error to normalize multiple boxes in predict sam1

Resolved: Error when normalizing the boxes during prediction with DINO.

Resolved: Error in using torch type bounding boxes in SAM2 prediction has been changed to numpy arrays

Resolved: Conversion of image/images to numpy for visualization of images and bounding boxes

Resolved: Duplicated compile c++ extensions for module _C of groundingdino

### Changed

Changed: Using box_convert from torchvision to convert bounding boxes from cxcywh format to xyxy format for SAM2

### Add

Add: Addition of code for displaying image, points, and bounding boxes for SAM2

Add: Post-processing of bounding boxes for SAM2 segmentation has been added

Add: The "TORCH_CUDNN_SDPA_ENABLED" rule which allows the use of special functions for SAM2