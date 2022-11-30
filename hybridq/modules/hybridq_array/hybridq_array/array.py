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
import numpy as np

__all__ = ['Array', 'array', 'asarray']


def _asarray(a: any):
    """
    If `a` is an `Array`, just return it. Otherwise, return `np.asarray(a)`.
    """
    return a if isinstance(a, Array) else np.asarray(a)


def array(obj,
          /,
          dtype: any = None,
          order: str = 'K',
          *,
          alignment: int = None,
          copy: bool = True):
    """
    Create `Array` from object `obj`.

    Parameters
    ----------
    obj: any
        Object to convert to `Array`.
    dtype: any, optional
        Type to use for the new `Array`. If not provided, infer from `obj`.
    order: str, optional
        Order to use for the new `Array`. If not provided, infer from `obj`.
    alignment: int, optional
        Alignment to use for the new `Array`. If not provided, and `obj` is
        `Array`, the alignment is inferred from `obj`. Otherwise, `alignment`
        is set to the default value.
    copy: bool, optional
        If `True`, always return a copy of `obj`. Otherwise, return a view if
        possible.
    """

    from .aligned_array import array, asarray, zeros_like

    # If obj is neither Array nor np.ndarray, convert it
    if not any(isinstance(obj, t) for t in (Array, np.ndarray)):
        obj = np.asarray(obj, dtype=dtype)

    # Get default values
    alignment = (obj.min_alignment if isinstance(obj, Array) else
                 _DEFAULTS['alignment']) if alignment is None else alignment

    # Convert dtype
    dtype = obj.dtype if dtype is None else np.dtype(dtype)

    # Check if dtype is complex
    _complex = np.iscomplexobj(dtype.type(1))

    # If dtype is provided, get real type
    _rtype = dtype.type(1).real.dtype

    # If casting complex -> float, warn
    if np.iscomplexobj(obj) and not _complex:
        from numpy import ComplexWarning
        from warnings import warn
        warn("Casting complex values to real "
             "discards the imaginary part.", ComplexWarning)

    # If obj is an instance of Array, just return it
    if isinstance(obj, Array):
        # Check copy
        _copy = True if obj.iscomplexobj() and _complex and (
            (obj.real.alignment >= alignment) ^
            (obj.imag.alignment >= alignment)) else copy

        # Get real part
        _r = (array if _copy else asarray)(obj.real,
                                           dtype=_rtype,
                                           order=order,
                                           alignment=alignment)
        if obj.iscomplexobj() and _complex:
            # Get imaginary part
            _i = (array if _copy else asarray)(obj.imag,
                                               dtype=_rtype,
                                               order=order,
                                               alignment=alignment)
        else:

            # Get imaginary part
            _i = zeros_like(_r, alignment=alignment) if _complex else None

        # Return
        return Array(buffer=(_r, _i, alignment))

    # If obj is an instance of np.ndarray, return Array using buffer
    else:
        # Get order
        _order = (
            'C' if obj.flags.c_contiguous else 'F') if order == 'K' else order

        if np.iscomplexobj(obj) and _complex:
            # Get real and imaginary parts
            _r = array(obj.real,
                       dtype=_rtype,
                       order=_order,
                       alignment=alignment)
            _i = array(obj.imag,
                       dtype=_rtype,
                       order=_order,
                       alignment=alignment)
        else:
            # Get real part
            _r = (array if copy else asarray)(obj.real,
                                              dtype=_rtype,
                                              order=_order,
                                              alignment=alignment)

            # Get imaginary part
            _i = zeros_like(_r, alignment=alignment) if _complex else None

        # Return
        return Array(buffer=(_r, _i, alignment))


def asarray(obj,
            /,
            dtype: any = None,
            order: str = 'K',
            *,
            alignment: int = None):
    """
    See Also
    --------
    hybridq.array.array
    """

    return array(obj, dtype=dtype, order=order, alignment=alignment, copy=False)


class Array:
    """
    `Array` is an array where the real and imaginary part are stored on
    different `AlignedArray`.

    Parameters
    ----------
    shape: tuple[int, ...], optional
        Shape of the new Array. Optional if `buffer` is provided.
    dtype: any, optional
        Type for `Array`.
    order: any, optional
        Order for `Array`.
    alignment: int, optional
        Alignment for `Array`.
    buffer: tuple[real: AlignedArray, imag: AlignedArray, alignment: int], optional
        If provided, `buffer` is used as real and imaginary part of `Array`.
        `buffer` is not copied.
    kwargs: optional
        Extra parameters to pass to `aligned_array.empty`.

    See Also
    --------
    AlignedArray
    """

    # Initialize __slots__
    __slots__ = ('_real', '_imag', '_min_alignment')

    def __init__(self,
                 shape: tuple[int, ...] = None,
                 dtype: any = _DEFAULTS['dtype'],
                 order: any = _DEFAULTS['order'],
                 alignment: int = _DEFAULTS['alignment'],
                 *,
                 buffer: tuple[AlignedArray, AlignedArray, int] = None,
                 **kwargs):

        # Allocate new data
        if buffer is None:
            from .aligned_array import empty

            # Get default alignment and dtype
            alignment = int(alignment)
            dtype = np.dtype(dtype)

            # Get real type
            rtype = dtype.type(1).real.dtype

            # Check if complex
            _complex = np.iscomplexobj(dtype.type(1))

            # Get shape
            try:
                shape = (int(shape),)
            except:
                shape = tuple(map(int, shape))

            # Set data
            self._real = empty(shape=shape,
                               dtype=rtype,
                               order=order,
                               alignment=alignment,
                               **kwargs)
            self._imag = empty(shape=shape,
                               dtype=rtype,
                               order=order,
                               alignment=alignment,
                               **kwargs) if _complex else None

            # Set min alignment
            self._min_alignment = alignment

        # Use buffer
        else:
            from .aligned_array import AlignedArray, isaligned

            # Set data
            self._real, self._imag, self._min_alignment = buffer

            # Check type real part
            if not isinstance(self.real, AlignedArray):
                raise TypeError(
                    f"'{type(self.real).__name__}' is not supported")

            # Underlying type must be real
            if np.iscomplexobj(self.real.dtype.type(1)):
                raise TypeError("Buffer must be real-valued vectors")

            # Check alignment real part
            if not self.real.isaligned(self.min_alignment):
                raise TypeError("Alignment is not consistent "
                                "with the provided buffer")

            # Check imaginary part
            if self.iscomplexobj():
                # Check type imaginary part
                if not isinstance(self.imag, AlignedArray):
                    raise TypeError(
                        f"'{type(self.imag).__name__}' is not supported")

                # Check alignment imaginary part
                if not self.imag.isaligned(self.min_alignment):
                    raise TypeError("Alignment is not consistent "
                                    "with the provided buffer")

                # Check dtype
                if self.real.dtype != self.imag.dtype:
                    raise TypeError("Types are not consistent")

                # Check shape
                if self.real.shape != self.imag.shape:
                    raise TypeError("Shapes are not consistent")

                # Check order
                if self.real.flags.c_contiguous != self.imag.flags.c_contiguous or self.real.flags.f_contiguous != self.imag.flags.f_contiguous:
                    raise TypeError("Orders are not consistent")

    ###########################################################################

    def __array__(self, dtype=None, order=None, **kwargs) -> numpy.ndarray:
        # Get complex
        _c = self.real + 1j * self.imag if self.iscomplexobj() else self.real

        # Return
        return np.asarray(_c, dtype=dtype, order=order, **kwargs)

    def __array_function__(self, func, types, args, kwargs):
        from .array_np import HANDLED_FUNCTIONS
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        #if not all(issubclass(t, Array) for t in types):
        #    return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def tolist(self) -> list[any, ...]:
        """
        Return `list` of `AlignedArray`.
        """
        return self.__array__().tolist()

    ###########################################################################

    def __len__(self):
        return self.shape[0]

    def __str__(self):
        _str = '(' + type(
            self).__name__ + f', min_alignment={self.min_alignment})'
        _str += '\n\nreal:\n-----\n' + str(self._real)
        if self.iscomplexobj():
            _str += '\n\nimag:\n-----\n' + str(self._imag)
        return _str

    def __repr__(self):
        _repr = '(' + type(
            self).__name__ + f', min_alignment={self.min_alignment})'
        _repr += '\n\nreal:\n-----\n' + repr(self._real)
        if self.iscomplexobj():
            _repr += '\n\nimag:\n-----\n' + repr(self._imag)
        return _repr

    def __getitem__(self, key):
        from .aligned_array import get_alignment

        # Get real and imaginary parts
        _real = self.real[key]
        _imag = self.imag[key] if self.iscomplexobj() else None

        if _real.ndim:
            # Get alignment
            _ra = min(self.min_alignment, get_alignment(_real))
            _rb = min(self.min_alignment,
                      get_alignment(_real)) if self.iscomplexobj() else _ra
            return Array(buffer=(_real, _imag, min(_ra, _rb)))
        else:
            return (_real if _imag is None else _real + 1j * _imag)

    def __setitem__(self, key, value):
        # Convert value to array
        value = value if isinstance(value, Array) else _asarray(value)

        # If self is real but value is complex, raise error
        if not self.iscomplexobj() and np.iscomplexobj(value):
            raise NotImplementedError(f"Cannot promote 'Array' of type "
                                      f"'{self.dtype}' to complex type "
                                      f"'{value.dtype}'")

        # Assign real part
        self.real[key] = value.real

        # Assign imag part
        if self.iscomplexobj():
            self.imag[key] = value.imag

        # Return
        return self

    ############################################################################

    def astype(self,
               dtype: any,
               order: str = 'K',
               alignment: int = None,
               copy: bool = True) -> Array:
        """
        Return `Array` with the given `dtype`, `order` and `alignment`.
        If `copy` is `False`, a view of `self` may be returned.
        """

        from .aligned_array import zeros

        # Get dtype
        dtype = np.dtype(dtype)

        # Get real type
        rtype = dtype.type(1).real.dtype

        # Get alignment
        alignment = self.min_alignment if alignment is None else int(alignment)

        # Get real part
        _r = self.real.astype(dtype=rtype,
                              order=order,
                              alignment=alignment,
                              copy=copy)

        # Get imaginary part
        if np.iscomplexobj(dtype.type(1)):
            _i = self.imag.astype(
                dtype=_r.dtype, order=order, alignment=alignment,
                copy=copy) if self.iscomplexobj() else zeros(
                    shape=_r.shape,
                    dtype=_r.dtype,
                    order='C' if _r.flags.c_contiguous else 'F',
                    alignment=alignment)
        else:
            if self.iscomplexobj():
                from numpy import ComplexWarning
                from warnings import warn
                warn(
                    "Casting complex values to real "
                    "discards the imaginary part.", ComplexWarning)
            _i = None

        # Return
        return Array(buffer=(_r, _i, alignment))

    def copy(self, alignment: int = None, order: str = 'K') -> Array:
        alignment = self.min_alignment if alignment is None else int(alignment)
        order = ('C' if self.real.flags.c_contiguous else
                 'F') if order == 'K' else _DEFAULTS['order']
        _r = self.real.copy(alignment=alignment, order=order)
        _i = self.imag.copy(alignment=alignment,
                            order=order) if self.iscomplexobj() else None
        return Array(buffer=(_r, _i, alignment))

    def ravel(self, *args, **kwargs):
        from .aligned_array import asarray
        _r = asarray(self.real.ravel(*args, **kwargs),
                     alignment=self.min_alignment)
        _i = asarray(
            self.imag.ravel(*args, **kwargs),
            alignment=self.min_alignment) if self.iscomplexobj() else None
        return Array(buffer=(_r, _i, self.min_alignment))

    def flatten(self, alignment: int = None, order: str = _DEFAULTS['order']):
        from .aligned_array import empty

        # Get default
        alignment = self.min_alignment if alignment is None else int(alignment)

        # Fill real buffer
        _r = empty(shape=self.size,
                   dtype=self.real.dtype,
                   order=order,
                   alignment=alignment)
        np.copyto(_r, self.real.ravel())

        # Fill imag buffer
        if self.iscomplexobj():
            _i = empty(shape=self.size,
                       dtype=self.imag.dtype,
                       order=order,
                       alignment=alignment)
            np.copyto(_i, self.imag.ravel())
        else:
            _i = None

        # Return
        return Array(buffer=(_r, _i, alignment))

    def conj(self):
        # If complex ...
        if self.iscomplexobj():
            from .aligned_array import empty_like

            # Get buffer
            _i = empty_like(self.imag, alignment=self.min_alignment)

            # Fill buffer
            _i[:] = -self.imag

        # .. otherwise ...
        else:
            _i = None

        # Return
        return Array(buffer=(self.real, _i, self.min_alignment))

    @property
    def T(self):
        from .aligned_array import asarray
        _r = asarray(self.real.T, alignment=self.min_alignment)
        _i = asarray(
            self.imag.T,
            alignment=self.min_alignment) if self.iscomplexobj() else None
        return Array(buffer=(_r, _i, self.min_alignment))

    def align(self, alignment: int, copy: bool = True):
        _r = self.real.align(alignment=alignment, copy=copy)
        _i = self.imag.align(alignment=alignment,
                             copy=copy) if self.iscomplexobj() else None
        return Array(buffer=(_r, _i, alignment))

    @property
    def alignment(self):
        return min(
            self.real.alignment,
            self.imag.alignment) if self.iscomplexobj() else self.real.alignment

    @property
    def min_alignment(self):
        return self._min_alignment

    ###########################################################################

    @property
    def dtype(self):
        # Get real type
        _rtype = self.real.dtype

        # If complex, return complex dtype
        if self.iscomplexobj():
            # Check imaginary time
            assert (_rtype == self.imag.dtype)

            # Return complex type
            return (_rtype.type([1]) + 1j * _rtype.type([1])).dtype

        # Otherwise, return real dtype
        else:
            return _rtype

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        if self.iscomplexobj():
            return self._imag
        else:
            from .aligned_array import zeros

            # Get params
            shape = self.real.shape
            dtype = self.real.dtype
            order = 'C' if self.real.flags.c_contiguous else 'F'
            alignment = self.min_alignment

            # Return
            return zeros(shape=shape,
                         dtype=dtype,
                         order=order,
                         alignment=alignment)

    @property
    def shape(self):
        return self.real.shape

    @property
    def size(self):
        from functools import reduce
        return reduce(lambda x, y: x * y, self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    def iscomplexobj(self) -> bool:
        return self._imag is not None
