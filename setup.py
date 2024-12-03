from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import os
import torch.utils.cpp_extension
import platform
import argparse
import glob


__version__ = '0.1.0'

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)

# Get PyTorch include paths
torch_include_dirs = torch.utils.cpp_extension.include_paths()

def parse_args():
    parser = argparse.ArgumentParser(description='Setup script for floating_point_kernels')
    parser.add_argument('--platform', choices=['arm', 'generic'], default='generic', help='Specify the target architecture for kernels')
    args, unknown = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + unknown  # Update sys.argv to remove parsed args
    return args

args = parse_args()

# Check if CUDA is available
cuda_available = torch.cuda.is_available()

ext_modules = []

if args.platform == 'generic':
    # Base extension for naive_matmul (always available)
    ext_modules.append(
        Extension(
            'functional',
            ['kernels/cpu/generic/matmul.cpp', 
             'kernels/cpu/generic/pack2b.cpp', 
             'kernels/cpu/generic/unpack2b.cpp',
             'kernels/cpu/generic/matmul2b.cpp',
             'kernels/cpu/generic/bindings.cpp'],
            include_dirs=[
                get_pybind_include(),
                get_pybind_include(user=True)
            ] + torch_include_dirs,
            language='c++',
            extra_compile_args=['-std=c++17', '-fvisibility=hidden'],
        )
    )




elif args.platform == 'arm':
    ext_modules.append(
        Extension(
            'functional',
            ['kernels/cpu/arm/matmul.cpp'],
            include_dirs=[
                get_pybind_include(),
                get_pybind_include(user=True)
            ] + torch_include_dirs,
            language='c++',
            extra_compile_args=['-std=c++17', '-fvisibility=hidden'],
        )
    )

    # Add CUDA extension if CUDA is available
if cuda_available:
    ext_modules.append(
        torch.utils.cpp_extension.CUDAExtension(
            'functional_cuda',
            ['kernels/gpu/matmul.cu'],
            include_dirs=[
                get_pybind_include(),
                get_pybind_include(user=True)
            ] + torch_include_dirs,
            extra_compile_args={'cxx': ['-std=c++17', '-fvisibility=hidden'],
                                'nvcc': ['-O2']},
        )
    )

setup(
    name='floating_point_kernels',
    version=__version__,
    author='Your Name',
    author_email='your_email@example.com',
    description='Floating Point CPU and GPU Kernels for PyTorch',
    long_description='',
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
    options={
        'build_ext': {
            'build_lib': os.path.join('ternify', 'tnn')
        }
    }
)
