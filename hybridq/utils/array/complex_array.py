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

    Parameters
    ----------
    real: list[number]
        Real part of `ComplexArray`.
    imag: list[number]
        Imaginary part of `ComplexArray`.
    copy: bool, optional
        If `True`, `real` and `imag` are copied instead of being referenced.
        (default: `False`)
    """

    def __init__(self,
                 real: list[number],
                 imag: list[number],
                 copy: bool = False):
        from hybridq.utils import isnumber

        # Convert to np.ndarray
        real = (np.array if copy else np.asarray)(real)
        imag = (np.array if copy else np.asarray)(imag)

        # Checks
        if real.dtype != imag.dtype:
            raise ValueError("'real' and 'imag' must have the same type")
        if type(real) != np.ndarray or np.iscomplexobj(real):
            raise ValueError("'real' must be a non-complex 'numpy.ndarray'")
        if type(imag) != np.ndarray or np.iscomplexobj(imag):
            raise ValueError("'imag' must be a non-complex 'numpy.ndarray'")
        if real.shape != imag.shape:
            raise ValueError("'real' and 'imag' must have the same shape")
        if real.flags.c_contiguous != imag.flags.c_contiguous:
            raise ValueError("'real' and 'imag' must have the same order")

        # Assign real and imaginary part
        self.__real = real
        self.__imag = imag

    @property
    def real(self) -> numpy.ndarray:
        return self.__real

    @property
    def imag(self) -> numpy.ndarray:
        return self.__imag

    def __len__(self) -> int:
        return len(self.real)

    # Convert to np.ndarray
    def __array__(self, dtype=None) -> numpy.ndarray:
        """
        Array to return when converted to `numpy.ndarray`.
        """
        # Get complex array
        c = self.real + 1j * self.imag

        # Return
        return c if dtype is None else c.astype(dtype)

    def __array_function__(self, func, types, args, kwargs) -> np.ndarray:
        """
        Using the dispatch mechanism, apply NumPy functions to `ComplexArray`.
        """
        # Check that 'func' has been implemented
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        # Note: this allows subclasses that don't override
        # __array_function__ to handle ComplexArray objects.
        if not all(
                any(issubclass(t, c)
                    for c in (self.__class__, np.ndarray))
                for t in types):
            return NotImplemented

        # Call handler
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __setitem__(self, key, value):
        self.real[key] = np.real(value)
        self.imag[key] = np.imag(value)

    def __getitem__(self, key) -> ComplexArray:
        # Get real and imaginary parts
        real = self.real[key]
        imag = self.imag[key]

        # If numbers, return complex
        if real.ndim == 0:
            return real + 1j * imag

        # Otherwise, return a view
        else:
            return ComplexArray(real, imag)

    @property
    def dtype(self) -> numpy.dtype:
        """
        Return type of `ComplexArray`.

        Returns
        -------
        numpy.dtype
            The type of `ComplexArray`.
        """
        return (1j * np.array([1], dtype=self.real.dtype)).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """
        Return shape of `ComplexArray`.

        Returns
        -------
        tuple[int, ...]
            The tuple of dimensions.
        """
        return self.real.shape

    @property
    def ndim(self) -> int:
        """
        Return number of dimensions of `ComplexArray`.

        Returns
        -------
        int
            The number of dimensions.
        """
        return self.real.ndim

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
        return np.gcd(get_alignment(self.real), get_alignment(self.imag))

    @property
    def T(self) -> ComplexArray:
        """
        Return the transposition of `ComplexArray` without copying.

        Returns
        -------
        ComplexArray
            The transposition of `ComplexArray`.
        """
        return ComplexArray(self.real.T, self.imag.T)

    def ravel(self) -> ComplexArray:
        """
        Return flattened `ComplexArray` without copyng.

        Returns
        -------
        ComplexArray
            The flattened `ComplexArray`.
        """
        return ComplexArray(self.real.ravel(), self.imag.ravel())

    def conj(self) -> ComplexArray:
        """
        Return the conjugation of `ComplexArray`.

        Returns
        -------
        ComplexArray
            The conjugated `ComplexArray`.
        """
        return ComplexArray(self.real, -self.imag)

    def flatten(self) -> ComplexArray:
        """
        Return flattened `ComplexArray`.

        Returns
        -------
        ComplexArray
            The flattened `ComplexArray`.
        """
        return ComplexArray(self.real.flatten(), self.imag.flatten())


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
    return a.real


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
    return a.imag


@implements(np.iscomplexobj)
def iscomplexobj(a: ComplexArray) -> bool:
    return True


@implements(np.iscomplex)
def iscomplex(a: ComplexArray) -> numpy.ndarray:
    return np.full(a.shape, True)


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
    return np.sum(a.real) + 1j * np.sum(a.imag)


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


@implements(np.linalg.norm)
def norm(a: ComplexArray) -> float:
    """
    Return the norm `ComplexArray`.

    Parameters
    ----------
    a: ComplexArray

    Returns
    -------
    float
        The norm of `ComplexArray`.
    """
    return np.sqrt(np.linalg.norm(a.real)**2 + np.linalg.norm(a.imag)**2)


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
    return ComplexArray(np.reshape(a.real, newshape=newshape, order=order),
                        np.reshape(a.imag, newshape=newshape, order=order))


@implements(np.vdot)
def vdot(a: ComplexArray, b: ComplexArray) -> complex:
    """
    Compute the scalar product of two `ComplexArray`s. The conjugate of `a`
    is taken.

    Parameters
    ----------
    a, b: ComplexArray

    Returns
    -------
    complex
        Scalar product between `a` and `b`.
    """
    if isinstance(a, ComplexArray) and isinstance(b, ComplexArray):
        return np.sum(a.real * b.real + a.imag * b.imag + 1j * a.real * b.imag -
                      1j * a.imag * b.real)
    else:
        return np.vdot(np.asarray(a), np.asarray(b))


def _shares_memory(a: ComplexArray | numpy.ndarray,
                   b: ComplexArray | numpy.ndarray, _func: callable) -> bool:
    """
    Check whetever `a` and `b` share memory.

    Parameters
    ----------
    a, b: ComplexArray | numpy.ndarray

    Returns
    -------
    bool
        `True` is `a` and `b` share memory, and `False` otherwise.
    """

    # Expand 'a' if a is a ComplexArray
    if isinstance(a, ComplexArray):
        return any(_shares_memory(x, b, _func=_func) for x in (a.real, a.imag))

    # Otherwise ...
    else:

        # Expand 'b' if b is a ComplexArray
        if isinstance(b, ComplexArray):
            return any(
                _shares_memory(a, x, _func=_func) for x in (b.real, b.imag))

        # Check if memory is shared
        else:
            return _func(a, b)


@implements(np.shares_memory)
def shares_memory(a: ComplexArray | numpy.ndarray,
                  b: ComplexArray | numpy.ndarray) -> bool:
    """
    Check whetever `a` and `b` share memory.

    Parameters
    ----------
    a, b: ComplexArray

    Returns
    -------
    bool
        `True` is `a` and `b` share memory, and `False` otherwise.
    """
    return _shares_memory(a, b, _func=np.shares_memory)


@implements(np.may_share_memory)
def may_share_memory(a: ComplexArray | numpy.ndarray,
                     b: ComplexArray | numpy.ndarray) -> bool:
    """
    Check whetever `a` and `b` may share memory.

    Parameters
    ----------
    a, b: ComplexArray | numpy.ndarray

    Returns
    -------
    bool
        `True` is `a` and `b` may share memory, and `False` otherwise.
    """
    return _shares_memory(a, b, _func=np.may_share_memory)
