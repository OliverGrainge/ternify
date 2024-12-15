from setuptools import setup, find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension

setup(
    name='ternify',  # Package name
    version='0.1',
    description='Ternary Neural Network Extensions',
    packages=find_packages(),  # Automatically find and include all packages
    package_dir={"": "."},  # Root directory
    ext_modules=[
        CppExtension(
            name='ternify.tnn.functional',  # Full module path
            sources=[
                'ternify/tnn/src/kernels/common/tlinear_forward.cpp',
                'ternify/tnn/src/kernels/common/primitives/tgemm.cpp',
                'ternify/tnn/src/kernels/common/pack2b.cpp',
                'ternify/tnn/src/kernels/common/unpack2b.cpp',
                'ternify/tnn/src/kernels/common/bindings.cpp',
            ],
            extra_compile_args=['-std=c++17', '-fopenmp'],
            extra_link_args=['-lgomp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
