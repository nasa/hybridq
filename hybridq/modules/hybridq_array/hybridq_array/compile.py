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

from subprocess import Popen, PIPE
from functools import lru_cache
from sysconfig import get_path
import logging
import os

from .defaults import _DEFAULTS, Default, parse_default

__all__ = []

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


@lru_cache
def get_config():
    """
    Get HybridQ core configuration.
    """

    # Get root of the package
    root_ = os.path.join(get_path('purelib'), 'hybridq_array/lib')

    # Get command for make
    cmd_ = f'make -C {root_} print_support'
    with Popen(cmd_.split(), stderr=PIPE, stdout=PIPE) as cmd_:
        # Get results
        out_, err_ = cmd_.communicate()

        # Raise if make has an error
        if cmd_.returncode:
            raise OSError(err_.decode())

    # Strip and split data
    out_ = out_.decode().strip().split('\n')

    # Initialize support
    support_ = {}

    # Get supported
    support_['openmp'] = next(filter(lambda x: ' OpenMP?' in x,
                                     out_)).split()[-1] == 'yes'
    support_['avx'] = next(filter(lambda x: ' AVX?' in x,
                                  out_)).split()[-1] == 'yes'
    support_['avx2'] = next(filter(lambda x: ' AVX-2?' in x,
                                   out_)).split()[-1] == 'yes'
    support_['avx512'] = next(filter(lambda x: ' AVX-512?' in x,
                                     out_)).split()[-1] == 'yes'
    support_['float16'] = next(filter(lambda x: ' Float16?' in x,
                                      out_)).split()[-1] == 'yes'
    support_['float128'] = next(filter(lambda x: ' Float128?' in x,
                                       out_)).split()[-1] == 'yes'

    # Return
    return support_


@lru_cache
def show_config():
    """
    Show HybridQ core configuration.
    """
    # Get root of the package
    root_ = os.path.join(get_path('purelib'), 'hybridq_array/lib')

    # Get command for make
    cmd_ = f'make -C {root_} print_support'
    with Popen(cmd_.split(), stderr=PIPE, stdout=PIPE) as cmd_:
        # Get results
        out_, err_ = cmd_.communicate()

        # Raise if make has an error
        if cmd_.returncode:
            raise OSError(err_.decode())

    return '\n'.join(
        filter(lambda x: x[0] == '#',
               out_.decode().strip().split('\n')))


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def compile_lib(target: str,
                libpath: str = Default,
                use_global_cache: bool = Default,
                **kwargs) -> None:
    """
    Compile the HybridQ C++ core library.
    """

    # If library cannot be loaded, try to compile
    if libpath is None:

        # Get root of the package
        root_ = os.path.join(get_path('purelib'), 'hybridq_array/lib')

        # Get cache folder
        _cache = os.path.join(
            os.path.expanduser('~'),
            '.cache/hybridq_array') if use_global_cache else os.path.join(
                os.getcwd(), '.hybridq_array')

        # Check if writable
        if not os.access(os.path.dirname(_cache), os.W_OK):
            _LOGGER.warning("Cannot create cache folder.")
            _LOGGER.warning("Cannot use C++ core.")

            # Return
            return

        # Create folder
        os.makedirs(_cache, exist_ok=True)

        # Try to compile
        _LOGGER.info("Try to compile C++ core to '%s' with parameters %s",
                     _cache, ', '.join(f'{k}={v}' for k, v in kwargs.items()))
        cmd_ = f'make -C {root_} {target} -j {os.cpu_count()} -e OUTPUT_PATH={_cache} '
        cmd_ += ' '.join(f'{k}={v}' for k, v in kwargs.items())
        with Popen(cmd_.split(), stderr=PIPE, stdout=PIPE) as cmd_:
            # Get results
            out_, err_ = cmd_.communicate()

            # Raise if make has an error
            if cmd_.returncode:
                raise OSError(err_.decode())

        # Log output
        if out_:
            _LOGGER.warning(out_.decode())

        if err_:
            _LOGGER.error(err_.decode())
