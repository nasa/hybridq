"""
Author: Salvatore Mandra (salvatore.mandra@nasa.gov)

Copyright © 2021, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The HybridQ: A Hybrid Simulator for Quantum Circuits platform is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from distutils.command.install import install as DistutilsInstall
from setuptools import find_packages, setup
from distutils.core import Extension
from os import path
import subprocess

# C++ Core
ext_modules = [
    Extension('hybridq_clifford_core',
              sources=['src/core.cpp'],
              include_dirs=[
                  x_[2:] for x_ in subprocess.Popen(
                      "python -m pybind11 --includes".split(),
                      stdout=subprocess.PIPE).communicate()[0].decode().split()
              ],
              language='c++',
              extra_compile_args=[
                  '-std=c++17', '-ffast-math', '-Ofast', '-Wall', '-Wpedantic',
                  '-Wfatal-errors'
              ]),
]

# Locate right path
here = path.abspath(path.dirname(__file__))

# Version
VERSION = '0.1.0'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    ext_modules=ext_modules,
    name='hybridq-clifford',
    version=VERSION,
    description='HybridQ-Clifford is a fast simulator of circuits based on '
    'the clifford expansion technique',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nasa/hybridq/modules/hybridq_clifford',
    author='Salvatore Mandrà',
    author_email='salvatore.mandra@nasa.gov',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8', 'Operating System :: Unix',
        'Operating System :: MacOS'
    ],
    python_requires='>=3.7',
    keywords=['clifford'],
    packages=find_packages(exclude=['docs', 'tests', 'tutorials']),
    install_requires=install_requires,
    project_urls={
        'Bug Reports': 'https://github.com/nasa/hybridq/issues',
        'Source': 'https://github.com/nasa/hybridq/modules/hybridq_clifford',
    },
)
