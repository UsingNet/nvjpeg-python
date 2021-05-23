#!/usr/bin/env python
import sys
import os
import platform
import glob
from setuptools import setup, find_packages, Extension
import numpy

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from distutils.core import setup, Extension

if platform.system() == 'Linux':
    if os.path.exists('/usr/src/jetson_multimedia_api'):
        # Jetson
        os.system('make lib_cuda')
        extension_nvjpeg = Extension('nvjpeg', 
            [
                'nvjpeg-python.cpp', 'src/jetson/JpegCoder.cpp',
                '/usr/src/jetson_multimedia_api/samples/common/classes/NvJpegDecoder.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvJpegEncoder.cpp',
                '/usr/src/jetson_multimedia_api/samples/common/classes/NvBuffer.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvElement.cpp',
                '/usr/src/jetson_multimedia_api/samples/common/classes/NvLogging.cpp', '/usr/src/jetson_multimedia_api/samples/common/classes/NvElementProfiler.cpp',
                '/usr/src/jetson_multimedia_api/argus/samples/utils/CUDAHelper.cpp'
            ], 
            ['include', '/usr/src/jetson_multimedia_api/argus/samples/utils', '/usr/src/jetson_multimedia_api/include', '/usr/src/jetson_multimedia_api/include/libjpeg-8b', numpy.get_include()], 
            [('JPEGCODER_ARCH', 'jetson')],
            library_dirs=['/usr/lib/aarch64-linux-gnu/tegra', 'build/lib'],
            libraries=['color_space', 'cudart', 'nvjpeg', 'cuda']
        )
    else:
        # x86 or x86_64
        extension_nvjpeg = Extension('nvjpeg', 
            ['nvjpeg-python.cpp', 'src/x86/JpegCoder.cpp'], 
            ['include', numpy.get_include()], 
            [('JPEGCODER_ARCH', 'x86')]
        )


setup(name='pynvjpeg',
    version='0.0.12',
    ext_modules=[extension_nvjpeg],
    author="Usingnet",
    author_email="developer@usingnet.com",
    license="MIT",
    description="Python interface for nvjpeg. Encode/Decode Jpeg with Nvidia GPU Hardware Acceleration.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UsingNet/nvjpeg-python",
    # packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    keywords=[
        "pynvjpeg",
        "nvjpeg",
        "jpeg",
        "jpg",
        "encode",
        "decode",
        "jpg encode",
        "jpg decode",
        "jpeg encode",
        "jpeg decode",
        "gpu",
        "nvidia"
    ],
    python_requires=">=3.6",
    project_urls={
        'Source': 'https://github.com/UsingNet/nvjpeg-python',
        'Tracker': 'https://github.com/UsingNet/nvjpeg-python/issues',
    },
    install_requires=['numpy>=1.17', 'wheel>=0.36.2']
)
