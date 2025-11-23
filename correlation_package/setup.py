# #!/usr/bin/env python3
# import os
# import torch

# from setuptools import setup, find_packages
# from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# cxx_args = ['-std=c++11']

# nvcc_args = [
#     '-gencode', 'arch=compute_50,code=sm_50',
#     '-gencode', 'arch=compute_52,code=sm_52',
#     '-gencode', 'arch=compute_60,code=sm_60',
#     '-gencode', 'arch=compute_61,code=sm_61',
#     '-gencode', 'arch=compute_70,code=sm_70',
#     '-gencode', 'arch=compute_70,code=compute_70'
# ]

# setup(
#     name='correlation_cuda',
#     ext_modules=[
#         CUDAExtension('correlation_cuda', [
#             'correlation_cuda.cc',
#             'correlation_cuda_kernel.cu'
#         ], extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args})
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     })
#!/usr/bin/env python3
import os
from setuptools import setup

# 直接导入 BuildExtension 和 CUDAExtension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ['-std=c++11']
nvcc_args = [
    '-O3',
    '-gencode', 'arch=compute_50,code=sm_50',
    '-gencode', 'arch=compute_52,code=sm_52',
    '-gencode', 'arch=compute_60,code=sm_60',
    '-gencode', 'arch=compute_61,code=sm_61',
    '-gencode', 'arch=compute_70,code=sm_70',
    '-gencode', 'arch=compute_75,code=sm_75',
    '-gencode', 'arch=compute_80,code=sm_80',
    '-gencode', 'arch=compute_86,code=sm_86',
    '-gencode', 'arch=compute_89,code=sm_89',
    '-gencode', 'arch=compute_90,code=sm_90',
    '-gencode', 'arch=compute_120,code=sm_120',
]

setup(
    name='correlation_cuda',
    ext_modules=[
        CUDAExtension(
            name='correlation_cuda',
            sources=[
                'correlation_cuda.cc',
                'correlation_cuda_kernel.cu'
            ],
            extra_compile_args={'cxx': cxx_args, 'nvcc': nvcc_args}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension  # ← 直接传入类，不是实例！
    },
    zip_safe=False
)