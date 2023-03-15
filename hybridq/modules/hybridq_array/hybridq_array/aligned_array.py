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
import numpy as np

from .defaults import _DEFAULTS, Default, parse_default
from .utils import isintegral

# Available methods
__all__ = [
    'isaligned', 'get_alignment', 'empty', 'zeros', 'ones', 'empty_like',
    'zeros_like', 'ones_like', 'array', 'asarray'
]


def isaligned(a: np.ndarray, /, alignment: int) -> bool:
    """
    Return `True` if `a` is aligned with `alignment`.

    Parameters
    ----------
    a: numpy.ndarray
        Array to check the alignment.
    alignment: int
        The desired alignment.

    Returns
    -------
    bool
        `True` is `a` is aligned to `alignment`, and `False` otherwise.
    """
    return (a.ctypes.data % alignment) == 0


def get_alignment(a: np.ndarray, /, log2_max_alignment: int = 32) -> int:
    """
    Get the largest alignment for `a`, up to `max_alignment`.

    Parameters
    ----------
    a: numpy.ndarray
        Array to get the alignment.
    max_alignment: int, optional
        Maximum alignment to check.

    Returns
    -------
    int
        The maximum alignment of `a`, up to `max_alignment`.
    """
    # Get best alignment
    return next((2**(x - 1)
                 for x in range(0, log2_max_alignment)
                 if a.ctypes.data % 2**x), 2**log2_max_alignment)


class AlignedArray(np.ndarray):
    """
    Extension of `numpy.ndarray` for aligned arrays.
    """

    @property
    def alignment(self) -> int:
        """
        Return alignment of `AlignedArray`. This is equivalent to
        `aligned_array.get_alignment(self)`.

        Returns
        -------
        int
            Alignment of `AlignedArray`.
        """
        return get_alignment(self)

    def isaligned(self, alignment: int) -> bool:
        """
        Check if `AlignedArray` alignment is compatible with `alignment`.

        Parameters
        ----------
        alignment: int
            Alignment to check.

        Returns
        -------
        bool
            `True` if `AlignedArray` alignment is compabile with `alignment`.
        """
        return isaligned(self, alignment)

    def align(self, alignment: int, copy: bool = True) -> AlignedArray:
        """
        Align `AlignedArray` to the given `alignment`. If `AlignedArray`
        alignment is compatible with `alignment`, a view is returned
        (unless `copy` is set to `True`).

        Parameters
        ----------
        alignment: int
            New alignment.
        copy: bool, optional
            If `True`, always return a copy of `AlignedArray`.

        Returns
        -------
        AlignedArray
            Aligned array. If `copy` is `False`, a view of `self` may be
            returned.
        """
        return (array if copy else asarray)(self, alignment=alignment)

    @parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
    def copy(self, order: str = 'K', alignment: int = Default) -> AlignedArray:
        """
        Return a copy of `AlignedArray`.

        Parameters
        ----------
        alignment: int, optional
            Alignment for the new `AlignedArray`.
        order: 'C' | 'K', optional
            Order for the new `AlignedArray`.

        Returns
        -------
        AlignedArray
            A copy of `self`.
        """

        # Get order
        if order == 'K':
            order = 'C' if self.flags.c_contiguous else 'F'

        # Create a new empty object
        buffer = empty(shape=self.shape,
                       dtype=self.dtype,
                       order=order,
                       alignment=alignment)

        # Copy
        np.copyto(buffer, self)

        # Return buffer
        return buffer

    def astype(self,
               dtype: any,
               order: str = 'K',
               alignment: int = None,
               copy: bool = True) -> AlignedArray:
        """
        Return an `AlignedArray` with the given `dtype`. If `copy` is `False`,
        a view may be returned.

        Parameters
        ----------
        dtype: any
            New type for `AlignedArray`.
        order: str, optional
            New order for `AlignedArray`.
        alignment: int, optional
            New alignment for `AlignedArray`.
        copy: bool, optional
            If `True`, a copy is always returned. Otherwise, a view is returned
            if possible.

        Returns
        -------
        AlignedArray
            An `AlignedArray` with the given `dtype`.
        """
        return array(self,
                     dtype=dtype,
                     order=order,
                     alignment=alignment,
                     copy=copy)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def empty(shape: any,
          dtype: any = Default,
          order: 'C' | 'F' = Default,
          *,
          alignment: int = Default,
          fill: any = None) -> AlignedArray:
    """
    Return an `numpy.ndarray` which is aligned to the given `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : 'C' | 'F', optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.
    fill: any, optional
        Use `fill` to fill array

    Returns
    -------
    AlignedArray
        The aligned array.

    See Also
    --------
    numpy
    """
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

    # Rescale size
    size = size * dtype.itemsize

    # Get new buffer
    buffer = np.empty((size + alignment,), dtype='uint8', order=order)

    # Get shift
    shift = alignment - buffer.ctypes.data % alignment

    # Re-align if needed, and view as correct type
    buffer = buffer[shift:shift + size].view(dtype)

    # Reshape
    buffer = np.reshape(buffer, shape, order=order)

    # Check alignment before returning
    assert buffer.ctypes.data % alignment == 0

    # Fill if needed
    if fill is not None:
        np.copyto(buffer, fill)

    # Return buffer
    return buffer.view(AlignedArray)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def zeros(shape: any,
          dtype: any = Default,
          order: 'C' | 'F' = Default,
          *,
          alignment: int = Default) -> AlignedArray:
    """
    Return an `numpy.ndarray` of zeros which is aligned to the given
    `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : 'C' | 'F', optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    AlignedArray
        The aligned array.

    See Also
    --------
    hybridq.utils.aligned.empty
    """
    return empty(shape=shape,
                 dtype=dtype,
                 order=order,
                 alignment=alignment,
                 fill=0)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def ones(shape: any,
         dtype: any = Default,
         order: 'C' | 'F' = Default,
         *,
         alignment: int = Default) -> AlignedArray:
    """
    Return an `numpy.ndarray` of ones which is aligned to the given `alignment`.

    Parameters
    ----------
    shape: any
        The shape of the new array.
    dtype: any, optional
        The type of the new array.
    order : 'C' | 'F', optional
        Memory layout.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.

    Returns
    -------
    AlignedArray
        The aligned array.

    See Also
    --------
    hybridq.utils.aligned.empty
    """
    return empty(shape=shape,
                 dtype=dtype,
                 order=order,
                 alignment=alignment,
                 fill=1)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def empty_like(a: np.ndarray, /, alignment: int = Default) -> AlignedArray:
    """
    Equivalent to `numpy.empty_like`.
    """

    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'

    # Return
    return empty(shape=shape, dtype=dtype, order=order, alignment=alignment)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def zeros_like(a: np.ndarray, /, alignment: int = Default) -> AlignedArray:
    """
    Equivalent to `numpy.zeros_like`.
    """

    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'

    # Return
    return zeros(shape=shape, dtype=dtype, order=order, alignment=alignment)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def ones_like(a: np.ndarray, /, alignment: int = Default) -> AlignedArray:
    """
    Equivalent to `numpy.ones_like`.
    """

    # Get params
    shape = a.shape
    dtype = a.dtype
    order = 'C' if a.flags.c_contiguous else 'F'

    # Return
    return ones(shape=shape, dtype=dtype, order=order, alignment=alignment)


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def array(a: any,
          /,
          dtype: any = None,
          order: 'C' | 'F' | 'A' | 'K' = 'K',
          *,
          alignment: int = Default,
          copy: bool = True,
          **kwargs) -> AlignedArray:
    """
    Return a copy of `a` which is aligned to the given `alignment`.

    Parameters
    ----------
    a: any
        Array to align.
    dtype: any, optional
        The type of the new array.
    order : 'C' | 'F' | 'A' | 'K', optional
        Memory layout.  'A' and 'K' depend on the order of input array a.
        'C' row-major (C-style),
        'F' column-major (Fortran-style) memory representation.
        'K' (keep) preserve input order.
        Defaults to 'C'.
    alignment: int, optional
        The required alignment.
    copy: bool, optional
        It copies `a` to a new array even if `a` is already aligned.
        (default: `True`)

    Returns
    -------
    AlignedArray
        The aligned array.

    See Also
    --------
    numpy, hybridq.utils.aligned.empty
    """
    # Store reference to the original array
    _a = a

    # Get array
    a = np.asarray(_a, dtype=dtype, order=order, **kwargs)

    # Check if new
    _new = not np.may_share_memory(a, _a)

    # Get dtype, size and alignment
    dtype = a.dtype
    shape = a.shape
    order = ('C' if a.flags.c_contiguous else 'F') if order == 'K' else order
    alignment = get_alignment(a) if alignment is None else int(alignment)

    # Check alignment is compatible
    if alignment % dtype.itemsize:
        raise ValueError(
            f"{dtype} is not compatible with 'alignment={alignment}'")

    # If new, check alignment and eventually return if already aligned
    if (_new or not copy) and isaligned(a, alignment=alignment):
        a = a.view(AlignedArray)
        return a

    # Get buffer
    buffer = empty(shape=shape,
                   dtype=dtype,
                   order=order,
                   alignment=alignment,
                   **kwargs)

    # Copy to buffer
    np.copyto(buffer, a)

    # Return buffer
    return buffer.view(AlignedArray)


def asarray(a: any,
            /,
            dtype: any = None,
            order: 'C' | 'F' | 'A' | 'K' = 'K',
            *,
            alignment: int = None) -> AlignedArray:
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
    order : 'C' | 'F' | 'A' | 'K', optional
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
    AlignedArray
        The aligned array.

    See Also
    --------
    numpy, hybridq.utils.aligned.empty
    """
    # If regular np.ndarray and alignment is not specified, use np.asarray
    if not isinstance(a, AlignedArray) and alignment is None:
        return np.asarray(a=a, dtype=dtype, order=order)

    # Otherwise, return AlignedArray
    return array(a, dtype=dtype, order=order, alignment=alignment, copy=False)
