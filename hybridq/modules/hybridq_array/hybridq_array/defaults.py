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

from distutils.sysconfig import get_python_lib
from autoray import register_function
from os import environ, path
import numpy as np

__all__ = []

# Define default parameters
_DEFAULTS = {
    'libpath':
        environ.get('HYBRIDQ_ARRAY_LIBPATH', None),
    'alignment':
        int(environ.get('HYBRIDQ_ARRAY_ALIGNMENT', 32)),
    'dtype':
        environ.get('HYBRIDQ_ARRAY_DTYPE', 'float32'),
    'order':
        environ.get('HYBRIDQ_ARRAY_ORDER', 'C'),
    'force_compilation':
        int(environ.get('HYBRIDQ_ARRAY_FORCE_COMPILATION', 1)),
    'use_global_cache':
        int(environ.get('HYBRIDQ_ARRAY_USE_GLOBAL_CACHE', 1)),
    'backend':
        environ.get('HYBRIDQ_ARRAY_BACKEND', 'numpy'),
    'raise_if_hcore_fails':
        int(environ.get('HYBRIDQ_ARRAY_RAISE_IF_HCORE_FAILS', 0))
}

# Register functions for autoray
register_function('hybridq_array', 'transpose', np.transpose)
register_function('hybridq_array', 'reshape', np.reshape)
register_function('hybridq_array', 'linalg.norm', np.linalg.norm)
register_function('hybridq_array', 'linalg.svd', np.linalg.svd)
register_function('hybridq_array', 'einsum', np.einsum)
