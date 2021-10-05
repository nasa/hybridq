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
from hybridq.utils.utils import isintegral
import numpy as np


def isaligned(a: np.ndarray, alignment: int) -> bool:
    """
    Return `True` if `a` is aligned with `alignment`.

    Parameters
    ----------
    a: np.ndarray
        Array to check the alignment.
    alignment: int
        The desired alignment.

    Returns
    -------
    bool
        `True` is `a` is aligned to `alignment`, and `False` otherwise.
    """
    return (a.ctypes.data % alignment) == 0


def get_alignment(a: np.ndarray, max_alignment: int = 128) -> int:
    """
    Get the largest alignment for `a`, up to `max_alignment`.

    Parameters
    ----------
    a: np.ndarray
        Array to get the alignment.
    max_alignment: int, optional
        Maximum alignment to check.

    Returns
    -------
    int
        The maximum alignment of `a`, up to `max_alignment`.
    """
    # Check max_alignment
    if bin(max_alignment).count('1') != 1:
        raise ValueError("'max_alignment' must be a power of 2.")

    # Get largest base
    b = int(np.log2(max_alignment))

    # Get best alignment
    return next(2**x for x in range(b, 0, -1) if (a.ctypes.data % 2**x) == 0)


def empty(shape: any,
          dtype: any = float,
          order: {'C', 'F'} = 'C',
          *,
          alignment: int = 16,
          **kwargs):
    """
    Return an `np.ndarray` which is aligned to the given `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : {'C', 'F'}, optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    np.ndarray
        The aligned array.

    See Also
    --------
    numpy
    """
    # Set defaults
    kwargs.setdefault('__gen__', np.empty)

    # Get dtype, size and alignment
    dtype = np.dtype(dtype)
    shape = int(shape) if isintegral(shape) else tuple(map(int, shape))
    size = np.prod(shape)
    order = str(order)
    alignment = int(alignment)

    # Check order
    if order not in list('FC'):
        raise ValueError(f"'{order}' is not a valid order.")

    # Check alignment is compatible
    if alignment % dtype.itemsize:
        raise ValueError(
            f"{dtype} is not compatible with 'alignment={alignment}'")

    # Get max_shift
    max_shift = alignment // dtype.itemsize

    # Get new buffer
    buffer = kwargs['__gen__']((size + max_shift,), dtype=dtype, order=order)

    # Get right shift
    shift = (alignment - (buffer.ctypes.data % alignment)) // dtype.itemsize
    assert (shift <= max_shift)

    # Re-align if needed
    buffer = buffer[shift:size + shift]

    # Check alignment
    assert (buffer.ctypes.data % alignment == 0)

    # Reshape
    buffer = np.reshape(buffer, shape, order=order)

    # Check alignment before returning
    assert (buffer.ctypes.data % alignment == 0)

    # Return buffer
    return buffer


def zeros(shape: any,
          dtype: any = float,
          order: {'C', 'F'} = 'C',
          *,
          alignment: int = 16,
          **kwargs):
    """
    Return an `np.ndarray` of zeros which is aligned to the given `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : {'C', 'F'}, optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    np.ndarray
        The aligned array.

    See Also
    --------
    hybridq.utils.aligned.empty
    """
    return empty(shape=shape,
                 dtype=dtype,
                 order=order,
                 alignment=alignment,
                 __gen__=np.zeros)


def ones(shape: any,
         dtype: any = float,
         order: {'C', 'F'} = 'C',
         *,
         alignment: int = 16,
         **kwargs):
    """
    Return an `np.ndarray` of ones which is aligned to the given `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : {'C', 'F'}, optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    np.ndarray
        The aligned array.

    See Also
    --------
    hybridq.utils.aligned.empty
    """
    return empty(shape=shape,
                 dtype=dtype,
                 order=order,
                 alignment=alignment,
                 __gen__=np.ones)


def empty_like(a: np.array) -> np.array:
    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'
    alignment = get_alignment(a)

    # Return
    return empty(shape=shape, dtype=dtype, order=order, alignment=alignment)


def zeros_like(a: np.array) -> np.array:
    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'
    alignment = get_alignment(a)

    # Return
    return zeros(shape=shape, dtype=dtype, order=order, alignment=alignment)


def ones_like(a: np.array) -> np.array:
    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'
    alignment = get_alignment(a)

    # Return
    return ones(shape=shape, dtype=dtype, order=order, alignment=alignment)


def array(a: any,
          dtype: any = None,
          order: {'C', 'F', 'A', 'K'} = 'K',
          *,
          alignment: int = 16,
          copy: bool = True,
          **kwargs) -> np.ndarray:
    """
    Return a copy of `a` which is aligned to the given `alignment`.

    Parameters
    ----------
    a: any
        Array to align.
    dtype: any, optional
        The type of the new array.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.
    copy: bool, optional
        It copies `a` to a new array even if `a` is already aligned.
        (default: `True`)

    Returns
    -------
    np.ndarray
        The aligned array.

    See Also
    --------
    numpy, hybridq.utils.aligned.empty
    """

    # Store reference to the original array
    _a = a

    # Get array
    a = np.asarray(_a, dtype=dtype, order=order)

    # Check if a new copy is created
    _new = a is not _a

    # Get dtype, size and alignment
    dtype = a.dtype
    shape = a.shape
    size = np.prod(shape)
    order = 'C' if a.flags.c_contiguous else 'F'
    alignment = int(alignment)

    # Check alignment is compatible
    if alignment % dtype.itemsize:
        raise ValueError(
            f"{dtype} is not compatible with 'alignment={alignment}'")

    # If new, check alignment and eventually return if already aligned
    if (_new or not copy) and isaligned(a, alignment=alignment):
        return a

    # Get max_shift
    max_shift = alignment // dtype.itemsize

    # If _new, resize
    if _new:
        # Resize memory
        a.resize(size + max_shift)

        # Reference to buffer
        buffer = a

        # Return to the orginal size
        a = a[:size]

    # Otherwise, get new buffer
    else:
        buffer = np.empty((size + max_shift,), dtype=dtype, order=order)

    # Get right shift
    shift = (alignment - (buffer.ctypes.data % alignment)) // dtype.itemsize
    assert (shift <= max_shift)

    # Re-align if needed
    buffer = buffer[shift:size + shift]

    # Reshape
    buffer = np.reshape(buffer, shape, order=order)

    # Check alignment
    assert (isaligned(buffer, alignment=alignment))

    # Copy if a was provided
    np.copyto(buffer, np.reshape(a, shape, order=order))

    # Return buffer
    return buffer


def asarray(a: any,
            dtype: any = None,
            order: {'C', 'F', 'A', 'K'} = 'K',
            *,
            alignment: int = 16,
            **kwargs) -> np.ndarray:
    """
    Convert `a` to an aligned array with the given `alignment`.

    Parameters
    ----------
    a: any, optional
        Array to align.
    shape: any, optional
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : {'C', 'F', 'A', 'K'}, optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'A' (any) means 'F' if `a` is Fortran contiguous, 'C' otherwise
        'K' (keep) preserve input order.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    np.ndarray
        The aligned array.

    See Also
    --------
    numpy, hybridq.utils.aligned.empty
    """

    return array(a=a, dtype=dtype, order=order, alignment=alignment, copy=False)
