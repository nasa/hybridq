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
    from .defaults import _DEFAULTS
    from .utils import load_library

    # If library cannot be loaded, try to compile
    if _DEFAULTS['libpath'] is None:
        from distutils.sysconfig import get_python_lib
        from subprocess import Popen, PIPE
        import os

        # Get root of the package
        _root = os.path.join(get_python_lib(), 'hybridq_array/lib')

        # Get cache folder
        _cache = os.path.join(
            os.path.expanduser('~'), '.cache/hybridq_array'
        ) if _DEFAULTS['use_global_cache'] else os.path.join(
            os.getcwd(), '.hybridq_array')

        # Check if writable
        if not os.access(os.path.dirname(_cache), os.W_OK):
            _LOGGER.warning("Cannot create cache folder.")
            _LOGGER.warning("Cannot use C++ core.")

            # Return
            return

        # Create folder
        os.makedirs(_cache, exist_ok=True)

        # Check
        assert (os.access(_cache, os.W_OK))

        # Try to compile
        _LOGGER.info("Try to compile C++ core to '%s'", _cache)
        try:
            Popen('make -C {} -j {} -e OUTPUT_PATH={}'.format(
                _root, os.cpu_count(), _cache).split(),
                  stderr=PIPE,
                  stdout=PIPE).communicate()
        except FileNotFoundError as e:
            _LOGGER.warning(e)

        # Try to load again
        if load_library('hybridq.so') is None or load_library(
                'hybridq_swap.so') is None:
            _LOGGER.warning("Cannot use C++ core.")


def compile_lib(target: str, **kwargs) -> None:
    """
    Compile the HybridQ C++ core library.
    """

    from .defaults import _DEFAULTS
    from .utils import load_library

    # If library cannot be loaded, try to compile
    if _DEFAULTS['libpath'] is None:
        from distutils.sysconfig import get_python_lib
        from subprocess import Popen, PIPE
        import os

        # Get root of the package
        _root = os.path.join(get_python_lib(), 'hybridq_array/lib')

        # Get cache folder
        _cache = os.path.join(
            os.path.expanduser('~'), '.cache/hybridq_array'
        ) if _DEFAULTS['use_global_cache'] else os.path.join(
            os.getcwd(), '.hybridq_array')

        # Check if writable
        if not os.access(os.path.dirname(_cache), os.W_OK):
            _LOGGER.warning("Cannot create cache folder.")
            _LOGGER.warning("Cannot use C++ core.")

            # Return
            return

        # Create folder
        os.makedirs(_cache, exist_ok=True)

        # Check
        assert (os.access(_cache, os.W_OK))

        # Try to compile
        _LOGGER.info("Try to compile C++ core to '%s'", _cache)
        _cmd = 'make -C {} {} -j {} -e OUTPUT_PATH={} '.format(
            _root, target, os.cpu_count(), _cache)
        _cmd += ' '.join(f'{k}={v}' for k, v in kwargs.items())
        _out, _err = Popen(_cmd.split(), stderr=PIPE, stdout=PIPE).communicate()

        # Raise if make has an error
        if len(_err):
            raise OSError(_err.decode())

        # Log output
        _LOGGER.warning(_out.decode())
