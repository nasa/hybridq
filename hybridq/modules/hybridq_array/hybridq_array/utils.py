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
from functools import lru_cache
from re import sub
import logging
import ctypes
import os

import numpy as np
from .defaults import _DEFAULTS, Default, parse_default

__all__ = [
    'isintegral', 'load_library', 'get_ctype', 'define_lib_fn', 'get_lib_fn'
]

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


def isintegral(x: any, /) -> bool:
    """
    Return `True` if `x` is integral. The test is done by converting the `x` to
    `int`.
    """
    # Check if 'x' can be converted to 'int'
    try:
        int(x)
    except (ValueError, TypeError):
        return False

    # If convertible, check that 'x' is not truncated when converted
    return int(x) == x


# pylint: disable=unsubscriptable-object
# Load library
@lru_cache
@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def load_library(libname: str,
                 prefix: iter[str, ...] = ('', 'lib', 'local/lib', 'usr/lib',
                                           'usr/local/lib'),
                 libpath: str | iter[str, ...] = Default,
                 use_global_cache: bool = Default):
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

    def _load(path):
        try:
            # Load library
            _lib = ctypes.cdll.LoadLibrary(path)

            # Log
            _LOGGER.info("Loaded '%s' located in '%s'", libname, path)

            # Return library
            return _lib

        # pylint: disable=bare-except
        except:
            return None

    # If libpath is None, set to current folder
    if libpath is None:
        libpath = [
            os.path.join(os.path.expanduser('~'), '.cache/hybridq_array')
            if use_global_cache else os.path.join(os.getcwd(), '.hybridq_array')
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
        map(lambda x: os.path.join(*x, libname), (
            (base, path) for base in libpath for path in prefix)))
                 if lib is not None), None)


# Define shorthand to get ctypes
@lru_cache
def get_ctype(x: np.dtype | str, /, pointer: bool = False):
    """
    Given `numpy.dtype`, return corresponding `ctype`.

    Parameters
    ----------
    x: numpy.dtype
        Type to convert to ctype.

    pointer: bool
        If `True`, return a pointer.

    Returns
    -------
    ctype
    """

    # Get c_type
    _c_types_map = {
        np.dtype(f'float{ctypes.sizeof(8*ctypes.c_float)}'):
            ctypes.c_float,
        np.dtype(f'float{ctypes.sizeof(8*ctypes.c_double)}'):
            ctypes.c_double,
        np.dtype(f'float{ctypes.sizeof(8*ctypes.c_longdouble)}'):
            ctypes.c_longdouble,
        np.dtype('int8'):
            ctypes.c_int8,
        np.dtype('int16'):
            ctypes.c_int16,
        np.dtype('int32'):
            ctypes.c_int32,
        np.dtype('int64'):
            ctypes.c_int64,
        np.dtype('uint8'):
            ctypes.c_uint8,
        np.dtype('uint16'):
            ctypes.c_uint16,
        np.dtype('uint32'):
            ctypes.c_uint32,
        np.dtype('uint64'):
            ctypes.c_uint64,
    }

    if isinstance(x, str):

        # Remove all spaces
        x = sub(r'\s+', '', x)

        # Check for special characters
        if sub(r'[A-Za-z0-9*]+', '',
               x) or x.count('*') > 1 or (x.count('*') == 1 and x[-1] != '*'):
            raise ValueError(f"'{x}' is not a valid type")

        # Check if pointer
        if x[-1] == '*':
            # Set pointer to trye
            pointer = True

            # Remove '*'
            x = x[:-1]

    # Get type
    _type = ctypes.c_void_p if x == 'void' else _c_types_map[np.dtype(x)]

    # Return pointer or type
    return ctypes.POINTER(_type) if pointer else _type


# Define shorthand for defining functions
@lru_cache
def define_lib_fn(lib, fname, restype, *argtypes):
    """
    Given a c-library handle `lib`, return function handle.
    """

    # Convert types
    restype = get_ctype(restype)
    argtypes = tuple(map(get_ctype, argtypes))

    # Create function
    func = getattr(lib, fname)
    func.argtypes = argtypes
    func.restype = restype

    # Return
    return func


# Return handle to the c-function to call using np.array's
@lru_cache
def get_lib_fn(lib, fname, restype, *argtypes):
    """
    Given a c-library handle `lib`, return python function.
    """

    # Convert type to string
    def _str(x, /):
        return sub(r'\s+', '', x if isinstance(x, str) else str(np.dtype(x)))

    # Convert all arguments to string
    restype = _str(restype)
    argtypes = tuple(map(_str, argtypes))

    # Get handle
    _fun = define_lib_fn(lib, fname, restype, *argtypes)

    # Get pointer
    def _get_pointer(x, t, /):

        # If type is 'void*', get pointer without conversion
        if t == 'void*':
            return x.ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))

        # Convert to the right type
        x = np.asarray(x, dtype=t[:-1] if t[-1] == '*' else t)

        # Return pointer
        return x.ctypes.data_as(get_ctype(x.dtype, pointer=t[-1]
                                          == '*')) if x.ndim else x

    # Define the function to call
    def _caller(*_argtypes):
        # Convert types and return result
        return _fun(*(map(_get_pointer, _argtypes, argtypes)))

    # Return caller
    return _caller
