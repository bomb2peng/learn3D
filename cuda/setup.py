from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []

ext_modules=[
    CUDAExtension('voxelization.cuda.voxelize_cuda', [
        'voxelization/cuda/voxelize_cuda.cpp',
        'voxelization/cuda/voxelize_cuda_kernel.cu',
        ]),
    ]

setup(
    description='PyTorch implementation of cuda voxelization routines from Kato',
    name='voxelization',
    packages=['voxelization', 'voxelization.cuda'],
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)