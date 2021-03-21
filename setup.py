#!/usr/bin/env python
import sys
import os
import glob
from setuptools import setup, find_packages, Extension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

from distutils.core import setup, Extension
setup(name='pynvjpeg',
    version='0.0.9',
    ext_modules=[Extension('nvjpeg', ['nvjpeg-python.c'])],
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
    install_requires=['numpy>=1.17']
)
