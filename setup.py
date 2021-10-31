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
from setuptools import setup, find_packages
from os import path, environ, cpu_count
from sys import stderr
import subprocess

# Check if the CPP should be installed
_use_hybridq_cpp_core = 'HYBRIDQ_DISABLE_CPP_CORE' not in environ


class MyInstall(DistutilsInstall):

    def run(self):
        if _use_hybridq_cpp_core:
            n_proc = environ['NPROC'] if 'NPROC' in environ else cpu_count()
            subprocess.run(['make', f'-j{n_proc}'])
            DistutilsInstall.run(self)
        else:
            DistutilsInstall.run(self)
            print(
                "C++ HybridQ core is disabled. optimize='evolution-hybrid' will "
                "not be available and substituted with optimize='evolution-einsum'.",
                file=stderr)


# Locate right path
here = path.abspath(path.dirname(__file__))

# Version
version = '0.8.0'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name='hybridq',
    version=version,
    description='Hybrid Simulator for Quantum Circuits',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/s-mandra/hybridq',
    author='Salvatore Mandrà',
    author_email = 'salvatore.mandra@nasa.gov',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only'
    ],
    python_requires='>=3.7, <3.9',
    keywords=['simulator quantum circuits', 'quantum computing'],
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=install_requires,
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/s-mandra/hybridq/issues',
        'Source': 'https://github.com/s-mandra/hybridq/',
    },
    include_package_data=True,
    scripts=['bin/hybridq', 'bin/hybridq-dm'],
    cmdclass={'install': MyInstall},
    **(dict(data_files=[['lib/', ['hybridq/utils/hybridq.so', 'hybridq/utils/hybridq_swap.so']]]) if _use_hybridq_cpp_core else {})
)
