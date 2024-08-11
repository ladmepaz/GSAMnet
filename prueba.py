import os
import glob
this_dir = os.path.dirname(os.path.abspath(__file__))
extensions_dir = os.path.join(this_dir, "src", "groundingdino", "models", "GroundingDINO", "csrc")
extensions_dir2 = os.path.join(this_dir, "src", "groundingdino", "models", "GroundingDINO", "csrc", "MsDeformAttn")


main_source = os.path.join(extensions_dir, "vision.cpp")
sources = glob.glob(os.path.join(extensions_dir2, "*.cpp"))
source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
    os.path.join(extensions_dir2, "*.cu")
)
print(main_source)
print(sources)

print(source_cuda)