import os
import sys
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

compute_capability = "sm_70"  # default
for arg in sys.argv:
    if arg.startswith("--compute-capability="):
        compute_capability = arg.split("=")[1]
        sys.argv.remove(arg)

class CustomBuildExt(BuildExtension):
    def build_extensions(self):
        for ext in self.extensions:
            for i, flag in enumerate(ext.extra_compile_args):
                if "sm_" in flag:
                    ext.extra_compile_args[i] = f'-gencode=arch=compute_{compute_capability[3:]},code={compute_capability}'
        super().build_extensions()

setup(
    name='adagrad_cuda',
    ext_modules=[
        CUDAExtension(
            name='adagrad_cuda',
            sources=['adagrad.cpp', 'adagrad_cuda.cu']
        )
    ],
    cmdclass={'build_ext': CustomBuildExt}
)
