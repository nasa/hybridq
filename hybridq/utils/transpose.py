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
from hybridq.utils.utils import load_library
from warnings import warn
import numpy as np
import ctypes


# Define shorthand for defining functions
def _define_function(lib, fname, restype, *argtypes):
    func = lib.__getattr__(fname)
    func.argtypes = argtypes
    func.restype = restype
    return func


# Load library
_lib_swap = load_library('hybridq_swap.so')

# If library isn't loaded properly, warn
if _lib_swap is None:
    warn("Cannot find C++ HybridQ core. "
         "Falling back to 'numpy.transpose' instead.")

# Get c_type
_c_types_map = {
    np.dtype('float32'): ctypes.c_float,
    np.dtype('float64'): ctypes.c_double,
    np.dtype('int32'): ctypes.c_int32,
    np.dtype('int64'): ctypes.c_int64,
    np.dtype('uint32'): ctypes.c_uint32,
    np.dtype('uint64'): ctypes.c_uint64,
}

# Get swap core
_swap_core = {
    np.dtype(t): _define_function(_lib_swap, f"swap_{t}", ctypes.c_int,
                                  ctypes.POINTER(_c_types_map[np.dtype(t)]),
                                  ctypes.POINTER(ctypes.c_uint32),
                                  ctypes.c_uint, ctypes.c_uint)
    for t in (t + str(b) for t in ['float', 'int', 'uint'] for b in [32, 64])
} if _lib_swap else None


def transpose(a: np.ndarray,
              axes: iter[int] = None,
              inplace: bool = False,
              backend: any = 'numpy',
              **kwargs) -> np.ndarray:
    """
    """

    # Set defaults
    kwargs.setdefault('force_numpy', False)
    kwargs.setdefault('raise_if_hcore_fails', False)

    # Select the correct backend
    if backend == 'numpy':
        import numpy as backend
    elif backend == 'jax':
        import jax.numpy as backend
    else:
        raise ValueError(f"Backend {backend} is not supported.")

    # If not provided, axis just invert 'a'
    if axes is None:
        axes = np.arange(a.ndim).astype(np.uint32)[::-1]

    # Convert axes to np.ndarray
    else:
        axes = np.asarray(axes, dtype=np.uint32)

    # Check all axes have been provided
    if len(axes) != a.ndim or any(x >= a.ndim for x in axes):
        raise IndexError("'axes' out of range.")

    # Get array
    def _get(x):
        # Copy reference
        _x = x

        # Get array
        x = np.asarray(x, order='C')

        # Return array and True if new
        return x, x is not _x

    # Get array
    a, _new = _get(a)

    # Check if HybridQ core can be called
    _use_hcore = not kwargs['force_numpy']
    # Check library has been loaded
    _use_hcore &= _lib_swap != None
    # Check type is available
    _use_hcore &= a.dtype in _c_types_map
    # Check if all axes for a have dimension 2
    _use_hcore &= a.shape == (2,) * a.ndim

    # For debug purposes
    if not _use_hcore and not kwargs['force_numpy'] and kwargs[
            'raise_if_hcore_fails']:
        raise AssertionError("Cannot use HybridQ core.")

    # Check if HybridQ swap can be called
    if _use_hcore:
        # Get number of ordered axes
        n_ord = next((i for i, x in enumerate(axes) if i != x), len(axes))

        # If already ordered, just return
        if n_ord == len(axes):
            return a

        # If sufficiently small, use HybridQ swap
        if 3 < len(axes) - n_ord <= 16:
            # Copy if needed
            if not inplace and not _new:
                a = np.array(a)

            # Get only unordered axes
            axes = axes[n_ord:]

            # Rescale axes
            axes = len(a.shape) - axes[::-1] - 1

            # Get pointer to a and axes
            a_ptr = a.ctypes.data_as(ctypes.POINTER(_c_types_map[a.dtype]))
            axes_ptr = axes.ctypes.data_as(
                ctypes.POINTER(_c_types_map[axes.dtype]))

            # Call library
            if _swap_core[a.dtype](a_ptr, axes_ptr, len(a.shape), len(axes)):
                raise ValueError("Something went wrong.")

            # Return array
            return a

        # Otherwise, fall back to numpy.transpose
        else:
            # Warn
            if not kwargs['force_numpy']:
                warn("Fallback to 'numpy.transpose'")

            return backend.transpose(a, axes)

    # Otherwise, fall back to numpy.transpose
    else:
        # Warn
        if not kwargs['force_numpy']:
            warn("Fallback to 'numpy.transpose'")

        # Return
        return backend.transpose(a, axes)

    # It should never arrive here
    assert (False)
