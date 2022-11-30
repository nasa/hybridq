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

from __future__ import annotations
from .defaults import _DEFAULTS
import logging

__all__ = ['isintegral', 'load_library', 'get_ctype', 'define_lib_fn']

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


def isintegral(x: any) -> bool:
    """
    Return `True` if `x` is integral. The test is done by converting the `x` to
    `int`.
    """
    try:
        int(x)
    except:
        return False
    else:
        return int(x) == x


# Load library
def load_library(libname: str,
                 prefix: iter[str, ...] = ('', 'lib', 'local/lib', 'usr/lib',
                                           'usr/local/lib'),
                 libpath: str | iter[str, ...] = _DEFAULTS['libpath']):
    """
    Load library `libname`.

    Parameters
    ----------
    libname: str
        Library name to load
    prefix: list[str, ...]
        A list of paths.
    libpath: str | iter[str, ...]
        Path(s) to look for the core library. If `None`, use the current
        folder + `/.hybridq_array/`.

    Returns
    -------
    CDLL | None:
        Return `CDLL` if the library can be loaded and `None` othewise.
    """
    from sys import base_prefix, exec_prefix
    from os import path, getcwd
    import ctypes

    def _load(path):
        try:
            # Load library
            _lib = ctypes.cdll.LoadLibrary(path)

            # Log
            _LOGGER.info("Loaded '%s' located in '%s'", libname, path)

            # Return library
            return _lib
        except:
            return None

    # If libpath is None, set to current folder
    if libpath is None:
        libpath = [
            path.join(path.expanduser('~'), '.cache/hybridq_array')
            if _DEFAULTS['use_global_cache'] else path.join(
                getcwd(), '.hybrudq_array')
        ]

    # Otherwise, use provided
    else:
        libpath = [libpath] if isinstance(libpath, str) else tuple(
            map(str, libpath))

    # Get prefix
    prefix = [prefix] if isinstance(prefix, str) else tuple(map(str, prefix))

    # Look for the library
    return next((lib for lib in map(
        _load,
        map(lambda x: path.join(*x, libname), (
            (base, path) for base in libpath for path in prefix)))
                 if lib is not None), None)


# Define shorthand to get ctypes
def get_ctype(x):
    from numpy import dtype
    import ctypes

    # Get c_type
    _c_types_map = {
        dtype(f'float{ctypes.sizeof(8*ctypes.c_float)}'):
            ctypes.c_float,
        dtype(f'float{ctypes.sizeof(8*ctypes.c_double)}'):
            ctypes.c_double,
        dtype(f'float{ctypes.sizeof(8*ctypes.c_longdouble)}'):
            ctypes.c_longdouble,
        dtype('int8'):
            ctypes.c_int8,
        dtype('int16'):
            ctypes.c_int16,
        dtype('int32'):
            ctypes.c_int32,
        dtype('int64'):
            ctypes.c_int64,
        dtype('uint8'):
            ctypes.c_uint8,
        dtype('uint16'):
            ctypes.c_uint16,
        dtype('uint32'):
            ctypes.c_uint32,
        dtype('uint64'):
            ctypes.c_uint64,
    }

    if isinstance(x, str):
        from re import sub

        # Remove all spaces
        _x = sub('\s+', '', x)

        # Check for special characters
        if sub(r'[A-Za-z0-9*]+', '',
               _x) or _x.count('*') > 1 or (_x.count('*') == 1 and
                                            _x[-1] != '*'):
            raise ValueError(f"'{x}' is not a valid type")

        # Get type
        _type = _c_types_map[dtype(_x[:-1] if _x[-1] == '*' else _x)]

        # Return pointer or type
        return ctypes.POINTER(_type) if _x[-1] == '*' else _type
    else:
        return _c_types_map[dtype(x)]


# Define shorthand for defining functions
def define_lib_fn(lib, fname, restype, *argtypes):
    # Convert types
    restype = get_ctype(restype)
    argtypes = tuple(map(get_ctype, argtypes))

    # Create function
    func = lib.__getattr__(fname)
    func.argtypes = argtypes
    func.restype = restype

    # Return
    return func
