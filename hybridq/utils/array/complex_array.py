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

# NumPy functions handled by ComplexArray
HANDLED_FUNCTIONS = {}


class ComplexArray:
    """
    Dispatched NumPy array representing an array of complex numbers.
    """

    def __init__(self,
                 object,
                 object_im=None,
                 *,
                 dtype: any = None,
                 copy: bool = True,
                 order: any = 'C',
                 alignment: int = 16) -> None:
        """
        Initialize `ComplexArray`.

        Parameters
        ----------
        object: array_like
            Array used as base for `ComplexArray`. It can be an array of either
            real of complex numbers. However, if `object_im` is specified,
            `object` must be an array of real numbers.
        object_im: array_like, optional
            Array used for the imaginary part of `ComplexArray`. It must be an
            array of real numbers.
        dype: any, optional
            Type for `ComplexArray`. It must be a valid complex type.
        copy: bool, optional
            Copy `object` and `object_im`. (default: `True`)
        order: any, optional
            Order in which data are stored in `ComplexArray`. (default: `C`)
        alignment: int, optional
            Alignment to use for `ComplexArray`. (default: 16)

        See Also
        --------
        numpy.ndarray
        """
        from functools import partial
        from hybridq.utils.aligned import array, asarray, zeros_like

        # Check that dtype is a complex type
        if dtype is not None and not np.iscomplexobj(np.array(1, dtype=dtype)):
            raise ValueError("'dtype' must be a valid complex type.")

        # Copy if needed
        _array = partial(array if copy else asarray,
                         alignment=alignment,
                         order=order)

        # Convert objects to aligned arrays
        object = _array(object, alignment=alignment)

        # Convert dtype
        dtype = object.dtype if dtype is None else np.dtype(dtype)

        # Get real type
        dtype_re = np.real(1j * np.array([1], dtype=dtype)).dtype

        # If object_im is None, split object in real and imaginary parts
        if object_im is None:
            if np.iscomplexobj(object):
                object_im = _array(np.imag(object), dtype=dtype_re)
                object_re = _array(np.real(object), dtype=dtype_re)
            else:
                object_re = _array(object, dtype=dtype_re)
                object_im = zeros_like(object_re)

        # Otherwise, convert to array
        else:
            # If 'object_im' is specified, 'object' must be an array of real numbers
            if np.iscomplexobj(object):
                raise ValueError(
                    "'object' cannot be an array of complex numbers if 'object_im' is specified"
                )

            # 'object_im' must be an array of real numbers
            if np.iscomplexobj(object_im):
                raise ValueError("'object_im' must be an array of real numbers")

            # Get real part
            object_re = _array(object, dtype=dtype_re)

            # Convert 'object_im' to array
            object_im = _array(object_im, dtype=dtype_re)

        # Check that real and imaginary parts have the same shape
        if object_re.shape != object_im.shape:
            raise ValueError(
                "Real and imaginary parts must have the same shape")

        # Assign real and imaginary part
        self.re = object_re
        self.im = object_im

    # Convert to np.ndarray
    def __array__(self, dtype=None) -> numpy.ndarray:
        """
        Array to return when converted to `numpy.ndarray`.
        """
        # Get complex array
        c = self.re + 1j * self.im

        # Return
        return c if dtype is None else c.astype(dtype)

    def __array_function__(self, func, types, args, kwargs) -> np.ndarray:
        """
        Using the dispatch mechanism, apply NumPy functions to `ComplexArray`.
        """
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle DiagonalArray objects.
        if not all(issubclass(t, self.__class__) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __setitem__(self, key, value):
        self.re[key] = np.real(value)
        self.im[key] = np.imag(value)

    def __getitem__(self, key):
        from hybridq.utils import isintegral
        if isintegral(key):
            return self.re[key] + 1j * self.im[key]
        else:
            return ComplexArray(self.re[key],
                                self.im[key],
                                dtype=self.dtype,
                                alignment=self.alignment)

    @property
    def dtype(self) -> numpy.dtype:
        """
        Return type of `ComplexArray`.

        Returns
        -------
        numpy.dtype
            The type of `ComplexArray`.
        """
        return (1j * np.array([1], dtype=self.re.dtype)).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return shape of `ComplexArray`.

        Returns
        -------
        tuple[int, ...]
            The tuple of dimensions.
        """
        return self.re.shape

    @property
    def ndim(self) -> int:
        """
        Return number of dimensions of `ComplexArray`.

        Returns
        -------
        int
            The number of dimensions.
        """
        return self.re.ndim

    @property
    def alignment(self) -> int:
        """
        Return alignment of `ComplexArray`.

        Returns
        -------
        int
            The alignment of `ComplexArray`.
        """
        from hybridq.utils.aligned import get_alignment
        return np.gcd(get_alignment(self.re), get_alignment(self.im))

    def conj(self) -> ComplexArray:
        """
        Return the conjugation of `ComplexArray`.

        Returns
        -------
        ComplexArray
            The conjugated `ComplexArray`.
        """
        return ComplexArray(self.re,
                            -self.im,
                            dtype=self.dtype,
                            alignment=self.alignment)

    @property
    def T(self) -> ComplexArray:
        """
        Return the transposition of `ComplexArray`.

        Returns
        -------
        ComplexArray
            The transposition of `ComplexArray`.
        """
        return ComplexArray(self.re.T,
                            self.im.T,
                            dtype=self.dtype,
                            alignment=self.alignment,
                            copy=False)

    def flatten(self) -> ComplexArray:
        """
        Return flattened `ComplexArray`.

        Returns
        -------
        ComplexArray
            The flattened `ComplexArray`.
        """
        return ComplexArray(self.re.flatten(),
                            self.im.flatten(),
                            dtype=self.dtype,
                            alignment=self.alignment,
                            copy=False)

    def ravel(self) -> ComplexArray:
        """
        Return flattened `ComplexArray` without copyng.

        Returns
        -------
        ComplexArray
            The flattened `ComplexArray`.
        """
        return ComplexArray(self.re.ravel(),
                            self.im.ravel(),
                            dtype=self.dtype,
                            alignment=self.alignment,
                            copy=False)


def implements(np_function):
    """
    Register an __array_function__ implementation for DiagonalArray objects.
    """

    # Generate decorator
    def decorator(func):
        HANDLED_FUNCTIONS[np_function] = func
        return func

    # Return decorator
    return decorator


@implements(np.real)
def real(a: ComplexArray) -> numpy.ndarray:
    """
    Return the real part of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    numpy.ndarray
        The real part of `ComplexArray`.
    """
    return a.re


@implements(np.imag)
def imag(a: ComplexArray) -> numpy.ndarray:
    """
    Return the imaginary part of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    numpy.ndarray
        The imaginary part of `ComplexArray`.
    """
    return a.im


@implements(np.sum)
def sum(a: ComplexArray) -> complex:
    """
    Return the sum all of the components of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    complex
        The sum of all the components of `ComplexArray`.
    """
    return np.sum(a.re) + 1j * np.sum(a.re)


@implements(np.prod)
def prod(a: ComplexArray) -> complex:
    """
    Return the product all of the components of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    complex
        The product of all the components of `ComplexArray`.
    """
    return np.prod(np.asarray(a))


@implements(np.conj)
def conj(a: ComplexArray) -> ComplexArray:
    """
    Return the conjugation of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    ComplexArray
        The conjugation of `ComplexArray`.
    """
    return a.conj()


@implements(np.linalg.norm)
def norm(a: ComplexArray) -> ComplexArray:
    """
    Return the transposition of `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    ComplexArray
        The transposition of `ComplexArray`.
    """
    return np.sqrt(np.linalg.norm(a.re)**2 + np.linalg.norm(a.im)**2)


@implements(np.vdot)
def vdot(a: ComplexArray, b: ComplexArray) -> complex:
    """
    Compute the scalar product of two `ComplexArray`s. The conjugate of `a` is
    taken.

    Parameters
    ----------
    a, b: ComplexArray

    Returns
    -------
    ComplexArray
        Scalar product between `a` and `b`.
    """
    return np.sum(a.re * b.re + a.im * b.im + 1j * a.re * b.im -
                  1j * a.im * b.re)


@implements(np.reshape)
def reshape(a: ComplexArray,
            newshape: tuple[int, ...],
            order: any = 'C') -> ComplexArray:
    """
    Reshape the `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray
        Array to reshape.
    newshape: tuple[int, ...]
        The new shape for `a`.
    order: any
        New order for the reshaped `ComplexArray`.

    Returns
    -------
    ComplexArray
        The reshaped `ComplexArray`.
    """
    return ComplexArray(np.reshape(a.re, newshape=newshape, order=order),
                        np.reshape(a.im, newshape=newshape, order=order),
                        dtype=a.dtype,
                        order=order,
                        alignment=a.alignment,
                        copy=False)


@implements(np.may_share_memory)
def reshape(a: ComplexArray, b: ComplexArray) -> bool:
    """
    Check whetever `a` and `b` may share memory.

    Parameters
    ----------
    a, b: ComplexArray

    Returns
    -------
    bool
        `True` is `a` and `b` may share memory, and `False` otherwise.
    """
    return any(
        np.may_share_memory(x, y) for x in (a.re, a.im) for y in (b.re, b.im))
