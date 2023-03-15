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
        _root = os.path.join(get_path('purelib'), 'hybridq_array/lib')

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
        _cmd = f'make -C {_root} {target} -j {os.cpu_count()} -e OUTPUT_PATH={_cache} '
        _cmd += ' '.join(f'{k}={v}' for k, v in kwargs.items())
        with Popen(_cmd.split(), stderr=PIPE, stdout=PIPE) as _cmd:
            # Get results
            _out, _err = _cmd.communicate()

            # Raise if make has an error
            if _cmd.returncode:
                raise OSError(_err.decode())

        # Log output
        if _out:
            _LOGGER.warning(_out.decode())

        if _err:
            _LOGGER.error(_err.decode())
