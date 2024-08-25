import os
import glob
import sys  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
import subprocess
import torch
from torch.utils.cpp_extension import CUDA_HOME,CppExtension,CUDAExtension,BuildExtension
from setuptools.command.install import install

from setuptools import setup, find_packages
link = "https://download.pytorch.org/whl/cu121"
REQUIRED_PACKAGES = [
    "numpy==1.26.4",
    "transformers==4.42.4",
    "huggingface_hub==0.23.5",
    "addict==2.4.0",
    "opencv-python==4.10.0.84",
    "pycocotools",
    "yapf",
    "timm",
    "supervision==0.22.0",
    "tqdm>=4.66.1",
    "scikit-learn",
    "hydra-core>=1.3.2",
    "iopath>=0.1.10",
    "ninja",
    "kaggle"
]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src", "groundingdino", "models", "GroundingDINO", "csrc")
    extension_dir_sam = os.path.join(this_dir,"src","segment_anything2","csrc")
    srcs_sam2 = [f"{this_dir}/src/segment_anything2/csrc/connected_components.cu"]

    main_source = os.path.join(extensions_dir, "vision.cpp")
    sources = glob.glob(os.path.join(extensions_dir, "**", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "**", "*.cu")) + glob.glob(
        os.path.join(extensions_dir, "*.cu")
    )
    sources = [main_source] + sources

    extension = CppExtension

    extra_compile_args = {"cxx": []}
    define_macros = []

    if CUDA_HOME is not None and (torch.cuda.is_available() or "TORCH_CUDA_ARCH_LIST" in os.environ):
        print("Compiling with CUDA")
        extension = CUDAExtension
        sources += source_cuda

        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-allow-unsupported-compiler"
        ]

    else:
        print("Compiling without CUDA")
        define_macros += [("WITH_HIP", None)]
        extra_compile_args["nvcc"] = []
        return None

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir] 

    ext_modules = [
        extension(
            "groundingdino._C",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        ),
        extension(
            "segment_anything2._C", 
            sources=srcs_sam2, 
            include_dirs=[extension_dir_sam],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args
        )
    ]

    return ext_modules

def build_extensions():


    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()
    setup(
        name="groundino_samnet",
        version="0.1.9",
        author="Wilhelm David Buitrago Garcia",
        url="https://github.com/WilhelmBuitrago/DiagAssistAI",
        description="A SAM model with GroundingDINO model",
        license=license,
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=REQUIRED_PACKAGES,
        ext_modules=get_extensions(),
        cmdclass={"build_ext": BuildExtension},
        python_requires='==3.10.12',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10"]
    )

if __name__ == "__main__":
    build_extensions()