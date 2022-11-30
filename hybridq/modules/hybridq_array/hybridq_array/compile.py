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
import logging

__all__ = []

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


def compile():
    from .utils import load_library

    # If library cannot be loaded, try to compile
    if load_library('hybridq.so') is None or load_library(
            'hybridq_swap.so') is None:
        from distutils.sysconfig import get_python_lib
        from os import cpu_count, makedirs
        from subprocess import Popen, PIPE
        from .defaults import _DEFAULTS
        from os import path

        # Get cache folder
        _cache = path.join(path.expanduser('~'), '.cache/hybridq_array/lib')

        # Get libpath
        _libpath = _DEFAULTS.get('libpath', '')

        # Check
        if (isinstance(_libpath, str) and
                _libpath != _cache) or (_cache not in list(_libpath)):
            _LOGGER.warning(
                "Cannot use auto-compilation if HYBRIDQ_ARRAY_LIBPATH "
                "has been provided.")
            _LOGGER.warning("Cannot use C++ core.")

        else:
            # Get root of the package
            _root = path.join(get_python_lib(), 'hybridq_array/lib')

            # Create folder
            makedirs(_cache, exist_ok=True)

            # Try to compile
            _LOGGER.info('Try to compile C++ core.')
            try:
                Popen('make -C {} -j {} -e OUTPUT_PATH={}'.format(
                    _root, cpu_count(), _cache).split(),
                      stderr=PIPE,
                      stdout=PIPE).communicate()
            except FileNotFoundError as e:
                _LOGGER.warning(e)

            # Try to load again
            if load_library('hybridq.so') is None or load_library(
                    'hybridq_swap.so') is None:
                _LOGGER.warning("Cannot use C++ core.")
