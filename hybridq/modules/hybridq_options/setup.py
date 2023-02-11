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

from os import path
from setuptools import setup, find_packages

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
    name='hybridq-options',
    version=VERSION,
    description='HybridQ-Options is a function decorator library to '
    'automatically retrieve default values. Default values can '
    'be updated on-the-fly without changing the function '
    'signature.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nasa/hybridq/modules/hybridq_options',
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
    python_requires='>=3.8',
    keywords=['options'],
    packages=find_packages(exclude=['docs', 'tests', 'tutorials']),
    install_requires=install_requires,
    project_urls={
        'Bug Reports': 'https://github.com/nasa/hybridq/issues',
        'Source': 'https://github.com/nasa/hybridq/modules/hybridq_options',
    },
)
