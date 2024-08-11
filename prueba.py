import os
import glob
import torch
this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "src", "groundingdino", "models", "GroundingDINO", "csrc")
extensions_dir2 = os.path.join(this_dir, "src", "groundingdino", "models", "GroundingDINO", "csrc", "MsDeformAttn")


main_source = os.path.join(extensions_dir, "vision.cpp")
sources = glob.glob(os.path.join(extensions_dir2, "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
    os.path.join(extensions_dir2, "*.cu")
)
print(torch.cuda.is_available())
print(sources)
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)