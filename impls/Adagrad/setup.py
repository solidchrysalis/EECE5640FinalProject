import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

archs = os.getenv("ARCHS", "-gencode=arch=compute_60,code=sm_60").split()

setup(
    name='adagrad_cuda',
    ext_modules=[
        CUDAExtension(
            name='adagrad_cuda',
            sources=['adagrad.cpp', 'adagrad_cuda.cu'],
            extra_cuda_cflags=archs
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
