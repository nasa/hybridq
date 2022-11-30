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

from setuptools import setup, find_packages
from os import path

# Define local modules
modules = {
    'hybridq-options': 'hybridq/modules/hybridq_options',
    'hybridq-decorators': 'hybridq/modules/hybridq_decorators',
    'hybridq-parallel': 'hybridq/modules/hybridq_parallel',
    'hybridq-array': 'hybridq/modules/hybridq_array'
}

# Locate right path
here = path.abspath(path.dirname(__file__))

# Get requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

# Update install_requires with local modules
install_requires.extend(
    [f'{_name} @ file://{here}/{_loc}' for _name, _loc in modules.items()])

# Run setup
setup(
    name='hybridq',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=install_requires,
)
