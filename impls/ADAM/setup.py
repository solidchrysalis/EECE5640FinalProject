import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

archs = os.getenv("ARCHS", "-gencode=arch=compute_60,code=sm_60").split()

setup(
    name='adam_cuda',
    ext_modules=[
        CUDAExtension(
            name='adam_cuda',
            sources=['adam.cpp', 'adam_cuda.cu'],
            extra_cuda_cflags=archs
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
