from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension
import os

def check_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

# Determine whether to build with CUDA support
use_cuda = check_cuda_available()

# Define sources and extensions
common_sources = [
    'kernels/common/tgemm_cpu.cpp',
    'kernels/common/pack2b_cpu.cpp',
    'kernels/common/unpack2b_cpu.cpp',
    'kernels/pack2b.cpp',
    'kernels/unpack2b.cpp',
    'kernels/tlinear_forward.cpp',
    'kernels/bindings.cpp',
]

cuda_sources = [
    'kernels/cuda/pack2b_cuda.cu',
    'kernels/cuda/unpack2b_cuda.cu',
]

extensions = []

# Add CPU extension
extensions.append(
    CppExtension(
        name='ternify.tnn.functional',
        sources=common_sources,
        include_dirs=[
            os.path.abspath('kernels'),
            os.path.abspath('kernels/common'),
        ],
        extra_compile_args=['-std=c++17', '-fopenmp'],
        extra_link_args=['-lgomp'],
    )
)

# Add CUDA extension if CUDA is available
if use_cuda:
    extensions.append(
        CUDAExtension(
            name='ternify.tnn.functional_cuda',
            sources=common_sources + cuda_sources,
            include_dirs=[
                os.path.abspath('kernels'),
                os.path.abspath('kernels/common'),
                os.path.abspath('kernels/cuda'),
            ],
            extra_compile_args={
                'cxx': ['-std=c++17', '-DUSE_CUDA', '-fopenmp'],  # Add USE_CUDA macro for the CPU part
                'nvcc': ['-DUSE_CUDA'],  # Add USE_CUDA macro for the CUDA part
            },
            extra_link_args=['-lgomp'],
        )
    )

setup(
    name='ternify',  # Package name
    version='0.1',
    description='Ternary Neural Network Extensions',
    packages=find_packages(),  # Automatically find and include all packages
    package_dir={"": "."},  # Root directory
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
