#!/usr/bin/env python3
import setuptools
import sys
import os
import glob

pwd = os.path.abspath(os.path.dirname(__file__))
os.system("cd '%s'; make" % (pwd,))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="nvjpeg",
    version="0.0.1",
    author="Usingnet",
    author_email="zengqinghui@usingnet.com",
    license="MIT",
    description="nvjpeg for python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UsingNet/nvjpeg-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta"
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Environment :: GPU :: NVIDIA CUDA :: 10.2",
        "Environment :: GPU :: NVIDIA CUDA :: 11.0",
    ],
    keywords="nvjpeg jpeg encode decode",
    python_requires=">=3.6",
    project_urls={
        'Source': 'https://github.com/UsingNet/nvjpeg-python',
        'Tracker': 'https://github.com/UsingNet/nvjpeg-python/issues',
    },
    data_files=[
        ('', ['nvjpeg-python.c', 'Makefile']),
        ('nvjpeg/lib', glob.glob('%s/nvjpeg/lib/*.so' % pwd))
    ],
    install_requires=['numpy']
)

