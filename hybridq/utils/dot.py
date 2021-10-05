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
from hybridq.utils.transpose import _swap_core
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
_lib_dot = load_library('hybridq.so')

# If library isn't loaded properly, warn
if _lib_dot is None:
    warn("Cannot find C++ HybridQ core. "
         "Falling back to 'numpy.dot' instead.")

# Get c_type
_c_types_map = {
    np.dtype('float32'): ctypes.c_float,
    np.dtype('float64'): ctypes.c_double,
}

# Get pack size
_log2_pack_size = _define_function(_lib_dot, "get_log2_pack_size",
                                   ctypes.c_uint32)() if _lib_dot else None

# Get dot core
_dot_core = {
    np.dtype(t): _define_function(_lib_dot, f"apply_U_{t}", ctypes.c_int,
                                  ctypes.POINTER(_c_types_map[np.dtype(t)]),
                                  ctypes.POINTER(_c_types_map[np.dtype(t)]),
                                  ctypes.POINTER(_c_types_map[np.dtype(t)]),
                                  ctypes.POINTER(ctypes.c_uint32),
                                  ctypes.c_uint, ctypes.c_uint)
    for t in (t + str(b) for t in ['float'] for b in [32, 64])
} if _lib_dot else None

# Get to_complex core
_to_complex_core = {
    np.dtype('complex' + str(2 * b)): _define_function(
        _lib_dot, f"to_complex{2*b}", ctypes.c_int,
        ctypes.POINTER(_c_types_map[np.dtype('float' + str(b))]),
        ctypes.POINTER(_c_types_map[np.dtype('float' + str(b))]),
        ctypes.POINTER(_c_types_map[np.dtype('float' + str(b))]), ctypes.c_uint)
    for b in [32, 64]
} if _lib_dot else None


def _maybe_view(a: array_like):
    """
    Determine if `a` is potentially a view of another array.
    """
    return a.base is not None


def to_complex(a: array_like, b: array_like):
    # Check that a and b have the same shape
    if a.shape != b.shape:
        raise ValueError("'a' and 'b' must have the same shape.")

    # Check that neither a nor b are complex objects
    if np.iscomplexobj(a) or np.iscomplexobj(b):
        raise ValueError("Both 'a' and 'b' must be real valued.")

    # Use faster C++ implementation
    if not _maybe_view(a) and not _maybe_view(
            b) and a.dtype == b.dtype and a.dtype in _c_types_map:
        # Get complex_type
        complex_type = (1j * np.array([0], dtype=a.dtype)).dtype

        # Get new array
        c = np.empty(a.shape, dtype=complex_type)

        # Get c_float_type
        c_float_type = _c_types_map[a.dtype]

        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(c_float_type))
        b_ptr = b.ctypes.data_as(ctypes.POINTER(c_float_type))
        c_ptr = c.ctypes.data_as(ctypes.POINTER(c_float_type))

        # Get size
        size = np.prod(a.shape)

        # Apply core
        _to_complex_core[complex_type](a_ptr, b_ptr, c_ptr, size)

    # Use slower numpy implementation
    else:
        c = a + 1j * b

    # Return complex
    return c


def to_complex_array(a: array_like):
    # Check that a is complex
    if not np.iscomplexobj(a):
        raise ValueError("'a' must be an array of complex numbers.")

    # Get real type
    float_type = np.real(np.array([0], dtype=a.dtype)).dtype

    # Get size
    size = np.prod(a.shape)

    # Split real and imaginary part
    return np.reshape(
        np.reshape(
            np.asarray(a, order='C').view(float_type), (np.prod(a.shape), 2)).T,
        (2,) + a.shape)


def dot(a: np.ndarray,
        b: np.ndarray,
        axes_b: iter[int] = None,
        b_as_complex_array: bool = False,
        inplace: bool = False,
        backend: any = 'numpy',
        **kwargs):
    """
    """

    # Select the correct backend
    if backend == 'numpy':
        import numpy as backend
    elif backend == 'jax':
        import jax.numpy as backend
    else:
        raise ValueError(f"Backend {backend} is not supported.")

    # set defaults
    kwargs.setdefault('out', None)
    kwargs.setdefault('force_numpy', False)
    kwargs.setdefault('raise_if_hcore_fails', False)
    kwargs.setdefault('swap_back', True)
    kwargs.setdefault('alignment', 32)

    # If axes are not provided, fall back to numpy.dot
    if axes_b is None:
        return backend.dot(a, b, out=kwargs['out'])

    # Get array
    def _get(x):
        # Copy reference
        _x = x

        # Get array
        x = np.asarray(x, order='C')

        # Return array and True if new
        return x, x is not _x

    # Convert to numpy.ndarray
    a, _new_a = _get(a)
    b, _new_b = _get(b)

    # Get number of axes
    a_ndim = a.ndim
    b_ndim = b.ndim - (1 if b_as_complex_array else 0)

    # Get shapes
    a_shape = np.asarray(a.shape)
    b_shape = np.asarray(b.shape[:len(b.shape) -
                                 (1 if b_as_complex_array else 0)])

    # Get axes
    axes_b = np.asarray(axes_b)

    # Get real_type and complex_type
    real_type = b.dtype if b_as_complex_array else np.real(
        np.array([1], dtype=b.dtype)).dtype
    complex_type = (1j * np.array([1], dtype=real_type)).dtype

    # Check right format for b if b_as_complex_array==True
    if b_as_complex_array:
        if b.shape[0] != 2:
            raise ValueError("'b' is in the wrong format.")
        if np.iscomplexobj(b):
            raise ValueError("'b' is expected to be real.")

    # Check axes
    if any(axes_b >= b_ndim):
        raise IndexError("Index not in 'b'")
    if a_shape[-1] != np.prod(b_shape[axes_b]):
        raise ValueError("'a' and 'b' are incompatible.")

    # Get positions
    pos = (b_ndim - axes_b[::-1] - 1).astype('uint32')

    # Get smallest swap size
    swap_size = 0 if np.all(pos >= _log2_pack_size) else next(
        k
        for k in range(_log2_pack_size, 2 * max(len(axes_b), _log2_pack_size) +
                       1)
        if sum(pos < k) <= k - _log2_pack_size)

    # Check if it is possible to use HybridQ core
    _use_hcore = not kwargs['force_numpy']
    # Check library has been loaded properly
    _use_hcore &= _lib_dot is not None
    # Check if real_type is supported
    _use_hcore &= real_type in _c_types_map
    # Check that a is a matrix
    _use_hcore &= a_ndim == 2 and a_shape[0] == a_shape[1]
    # Check that the system isn't too small
    _use_hcore &= b_ndim >= 6 and (b_ndim - len(axes_b)) >= _log2_pack_size
    # Check that all axes for b have dimension 2
    _use_hcore &= all(x == 2 for x in b_shape)
    # Check that not too many axes are used
    _use_hcore &= len(axes_b) <= 10
    # Check that swap is not too large
    _use_hcore &= swap_size <= 12

    # For debug purposes
    if not _use_hcore and not kwargs['force_numpy'] and kwargs[
            'raise_if_hcore_fails']:
        raise AssertionError("Cannot use HybridQ core.")

    # Check if the HybridQ C++ core can be used
    if _use_hcore:
        from hybridq.utils.aligned import array, isaligned

        # Convert
        if a.dtype != complex_type:
            warn(f"'a' is recast to '{complex_type}' to match 'b'.")
            a = a.astype(complex_type)

        # Convert and align
        if b_as_complex_array:
            b = array(b,
                      copy=not (inplace or _new_b),
                      alignment=kwargs['alignment'])
        else:
            b = array([np.real(b), np.imag(b)],
                      dtype=real_type,
                      alignment=kwargs['alignment'])

        # Split in real and imaginary part
        b_re = b[0]
        b_im = b[1]

        # Check
        assert (b_re.ctypes.data % 32 == 0)
        assert (b_im.ctypes.data % 32 == 0)

        # Get pointers
        a_ptr = a.ctypes.data_as(ctypes.POINTER(_c_types_map[real_type]))
        b_re_ptr = b_re.ctypes.data_as(ctypes.POINTER(_c_types_map[real_type]))
        b_im_ptr = b_im.ctypes.data_as(ctypes.POINTER(_c_types_map[real_type]))

        # Swap real and imaginary part if required
        if swap_size:
            # Get swap transposition
            swap_pos = np.concatenate(
                (np.setdiff1d(range(swap_size),
                              pos), np.intersect1d(range(swap_size),
                                                   pos))).astype('uint32')
            swap_pos_ptr = swap_pos.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32))
            swap_pos_inv = np.argsort(swap_pos).astype('uint32')
            swap_pos_inv_ptr = swap_pos_inv.ctypes.data_as(
                ctypes.POINTER(ctypes.c_uint32))

            # Transpose real and imaginary part
            if _swap_core[real_type](b_re_ptr, swap_pos_ptr, b_ndim, swap_size):
                raise ValueError("Something went wrong with '_swap_core'.")
            if _swap_core[real_type](b_im_ptr, swap_pos_ptr, b_ndim, swap_size):
                raise ValueError("Something went wrong with '_swap_core'.")

            # Swap positions
            pos = np.array(
                [swap_pos_inv[x] if x < swap_size else x for x in pos],
                dtype='uint32')

        # Get pointer
        pos_ptr = pos.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))

        # Call HybridQ C++ core
        if _dot_core[real_type](b_re_ptr, b_im_ptr, a_ptr, pos_ptr, b_ndim,
                                len(pos)):
            raise ValueError('Something went wrong.')

        # Swap back
        if swap_size and kwargs['swap_back']:
            # Transpose real and imaginary part
            if _swap_core[real_type](b_re_ptr, swap_pos_inv_ptr, b_ndim,
                                     swap_size):
                raise ValueError("Something went wrong with '_swap_core'.")
            if _swap_core[real_type](b_im_ptr, swap_pos_inv_ptr, b_ndim,
                                     swap_size):
                raise ValueError("Something went wrong with '_swap_core'.")

        # Get result
        res = b if b_as_complex_array else b[0] + 1j * b[1]

        # Get the right transposition to use in `hybridq.utils.transpose`
        if swap_size and not kwargs['swap_back']:
            tr = list(range(b_ndim - swap_size)) + (
                b_ndim - swap_pos_inv[::-1] - 1).tolist()

        # Return
        return res if kwargs['swap_back'] is True else (
            res, tr if swap_size else None)

    # Otherwise, fallback to numpy.dot
    else:
        # Warn
        if not kwargs['force_numpy']:
            warn("Fallback to 'numpy.dot'")

        # Convert to complex array if needed
        if b_as_complex_array:
            b = backend.reshape(b[0] + 1j * b[1], b_shape)

        # Get right transposition
        pos = axes_b.tolist() + [x for x in range(b_ndim) if x not in axes_b]

        # Reshape and transpose
        b = backend.reshape(backend.transpose(b, pos), (np.prod(
            b_shape[axes_b]), np.prod(b_shape) // np.prod(b_shape[axes_b])))

        # Invert posisions
        pos = [pos.index(x) for x in range(len(pos))]

        # Apply a.dot(b), reshape to the original shape and transpose back
        b = backend.transpose(backend.reshape(backend.dot(a, b), b_shape), pos)

        # Return
        return backend.array([backend.real(b), backend.imag(b)
                             ]) if b_as_complex_array else b

    # It should never reach here
    assert (False)
