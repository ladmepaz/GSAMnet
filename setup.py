import os
import glob
import sys  
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
import subprocess
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
    "ninja"
]


def get_torch(): 
    TORCH_AVAILABLE = False
    env = os.environ.copy()
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", link],env=env)
        TORCH_AVAILABLE = True
        os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'
    except Exception as e:
        raise RuntimeError(f"Error downloading torch. Ensure that {link} is functional. ERROR: {str(e)}")
    
    try:
        subprocess.check_call([sys.executable,"-m","pip","install","hydra-core","--upgrade"],env=env)
    except Exception as e:
        raise RuntimeError(f"Error downloading hydra-cores: {str(e)}")
    
    return TORCH_AVAILABLE
def get_extensions(TORCH_AVAILABLE: bool):
    if TORCH_AVAILABLE:
        import torch
        from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

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


def parse_requirements(fname="requirements.txt", with_version=True):
    """Parse the package dependencies listed in a requirements file but strips
    specific versioning information.

    Args:
        fname (str): path to requirements file
        with_version (bool, default=False): if True include version specs

    Returns:
        List[str]: list of requirements items

    CommandLine:
        python -c "import setup; print(setup.parse_requirements())"
    """
    import re
    import sys
    from os.path import exists

    require_fpath = fname

    def parse_line(line):
        """Parse information from a line in a requirements text file."""
        if line.startswith("-r "):
            # Allow specifying requirements in other files
            target = line.split(" ")[1]
            for info in parse_require_file(target):
                yield info
        else:
            info = {"line": line}
            if line.startswith("-e "):
                info["package"] = line.split("#egg=")[1]
            elif "@git+" in line:
                info["package"] = line
            else:
                # Remove versioning from the package
                pat = "(" + "|".join([">=", "==", ">"]) + ")"
                parts = re.split(pat, line, maxsplit=1)
                parts = [p.strip() for p in parts]

                info["package"] = parts[0]
                if len(parts) > 1:
                    op, rest = parts[1:]
                    if ";" in rest:
                        # Handle platform specific dependencies
                        # http://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-platform-specific-dependencies
                        version, platform_deps = map(str.strip, rest.split(";"))
                        info["platform_deps"] = platform_deps
                    else:
                        version = rest  # NOQA
                    info["version"] = (op, version)
            yield info

    def parse_require_file(fpath):
        with open(fpath, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    for info in parse_line(line):
                        yield info

    def gen_packages_items():
        if exists(require_fpath):
            for info in parse_require_file(require_fpath):
                parts = [info["package"]]
                if with_version and "version" in info:
                    parts.extend(info["version"])
                if not sys.version.startswith("3.4"):
                    # apparently package_deps are broken in 3.4
                    platform_deps = info.get("platform_deps")
                    if platform_deps is not None:
                        parts.append(";" + platform_deps)
                item = "".join(parts)
                yield item

    packages = list(gen_packages_items())
    return packages


def build_extensions():
    TORCH_AVAILABLE = get_torch()
    if TORCH_AVAILABLE:
        from torch.utils.cpp_extension import BuildExtension

    with open("LICENSE", "r", encoding="utf-8") as f:
        license = f.read()
    setup(
        name="groundino_samnet",
        version="0.1.0",
        author="Wilhelm David Buitrago Garcia",
        url="https://github.com/WilhelmBuitrago/DiagAssistAI",
        description="A SAM model with GroundingDINO model",
        license=license,
        package_dir={"": "src"},
        packages=find_packages(where="src"),
        install_requires=REQUIRED_PACKAGES,
        ext_modules=get_extensions(TORCH_AVAILABLE=TORCH_AVAILABLE),
        cmdclass={"build_ext": BuildExtension},
        python_requires='==3.10.12',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.10"]
    )

if __name__ == "__main__":
    build_extensions()