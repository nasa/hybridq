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


def _type(x):
    try:
        float(x)
    except:
        return type(x)
    else:
        return float


class _Wrapper:
    """
    Wrapper used to sort most of the iterables.
    """

    def __init__(self, v):
        # Only hashable objects are allowed
        if not getattr(v, '__hash__', None):
            raise ValueError("Only hashable objects are allowed.")

        # Assign
        self.__v = v

    @property
    def v(self):
        return self.__v

    def __str__(self):
        return self.v.__str__()

    def __repr__(self):
        return self.v.__repr__()

    def __hash__(self):
        return self.v.__hash__()

    def __eq__(self, other: any):
        # Get value if Wrapper
        other = other.v if isinstance(other, _Wrapper) else v

        # Return if equal
        return self.v == other

    def __lt__(self, other: any):
        # Get value if Wrapper
        other = other.v if isinstance(other, _Wrapper) else v

        # If they are the same, just return False
        if self.v == other:
            return False

        # Try directly
        try:
            return self.v < other

        # Try alternatives
        except:

            # If types differ, sort accordingly to the type
            if type(self.v) != type(other):
                return str(type(self.v)) < str(type(other))

            # If the types are the same, sort accordingly to
            # their representation
            try:
                # Get the two representations
                _r1 = self.v.__repr__()
                _r2 = other.__repr__()

                # Ther two representations should be different as
                # we already checked that the two objects differ
                if _r1 != _r2:
                    return _r1 < _r2

            except:
                pass

            # Raise an Error if nothing works
            raise TypeError(f"'<' not supported.")

    def __le__(self, other):
        return self == other or self < v

    def __ge__(self, other):
        return not self < other

    def __gt__(self, other):
        return not self <= other


def isintegral(x: any):
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


def isnumber(x: any):
    """
    Return `True` if `x` is integral. The test is done by converting the `x` to
    `int`.
    """
    try:
        float(x)
    except:
        return False
    else:
        return True


def isdict(x: any,
           key_type: callable = lambda x: x,
           value_type: callable = lambda x: x,
           *,
           return_dict: bool = False):
    """
    Check if `x` is a dictionary. If `key_type` (`value_type`) is provided,
    check if every key (value) is convertible to `key_type` (`value_type`).

    Parameters
    ----------
    x: any
        Object to convert
    key_type: callable, optional
        If provided, convert keys using `key_type`.
    value_type: callable, optional
        If provided, convert values using `value_type`.
    return_dict: bool, optional
        If `True`, the converted `dict` is returned (instead of
        `True`/`False`), and an error is raised if `x` cannot be converted.

    Returns
    -------
    {bool, dict}
        If `return_dict` is `False`, `True` is returned if `x` can be converted
        to `dict` and `False` otherwise. If `return_dict` is `True`, the
        converted dictionary is returned and an error is raised if impossible
        to convert.
    """

    try:
        # Try to convert to dict
        x = {key_type(k): value_type(v) for k, v in dict(x).items()}

        # Return
        return x if return_dict else True

    except Exception as e:
        # If 'isdict' should return the dict, raise instead
        if return_dict:
            raise e

        # Otherwise, just return False
        else:
            return False


def islist(x: any,
           value_type: any = lambda x: x,
           *,
           list_type: type = list,
           return_list: bool = False):
    """
    Check if `x` is a list. If `value_type` is provided, check if every value
    is convertible to `value_type`.

    Parameters
    ----------
    x: any
        Object to convert
    value_type: callable, optional
        If provided, convert values using `value_type`.
    list_type: type, optional,
        Use `list_type` as list if provided (`list_type=list` by default).
    return_list: bool, optional
        If `True`, the converted `list` is returned (instead of
        `True`/`False`), and an error is raised if `x` cannot be converted.

    Returns
    -------
    {bool, list_type}
        If `return_list` is `False`, `True` is returned if `x` can be converted
        to `list_type` and `False` otherwise. If `return_list` is `True`, the
        converted list is returned and an error is raised if impossible to
        convert.
    """

    try:
        # Try to convert to list
        x = list_type(map(value_type, x))

        # Return
        return x if return_list else True

    except Exception as e:
        # If 'islist' should return the list, raise instead
        if return_list:
            raise e

        # Otherwise, just return False
        else:
            return False


def to_dict(x: any, key_type: any = lambda x: x, value_type: any = lambda x: x):
    """
    Convert `x` to a dictionary. If `key_type` (`value_type`) is provided,
    convert every key (value) to `key_type` (`value_type`).

    Parameters
    ----------
    x: any
        Object to convert
    key_type: callable, optional
        If provided, convert keys using `key_type`.
    value_type: callable, optional
        If provided, convert values using `value_type`.

    Returns
    -------
    dict
        The converted dictionary is returned and an error is raised if
        impossible to convert.
    """
    return isdict(x=x,
                  key_type=key_type,
                  value_type=value_type,
                  return_dict=True)


def to_list(x: any, value_type: any = lambda x: x, list_type: type = list):
    """
    Convert `x` to list. If `value_type` is provided, convert every value to
    `value_type`.

    Parameters
    ----------
    x: any
        Object to convert
    value_type: callable, optional
        If provided, convert values using `value_type`.
    list_type: type, optional,
        Use `list_type` as list if provided (`list_type=list` by default).

    Returns
    -------
    list_type
        The converted list is returned and an error is raised if impossible to
        convert.
    """
    return islist(x=x,
                  value_type=value_type,
                  list_type=list_type,
                  return_list=True)


def sort(iterable, *, key=None, reverse=False):
    """
    Sort heterogeneous list.

    """

    return sorted(iterable,
                  key=lambda x: _Wrapper(x if key is None else key(x)),
                  reverse=reverse)


def argsort(iterable, *, key=None, reverse=False):
    """
    Argsort heterogeneous list.
    """

    return [
        x for _, x in sort((
            (y if key is None else key(y), x) for x, y in enumerate(iterable)),
                           key=lambda x: x[0],
                           reverse=reverse)
    ]


def svd(a, axes: iter[int], sort: bool = False, atol: float = 1e-8, **kwargs):
    """
    Return the SVD of `a` by splitting it accordingly to `axes`.

    Parameters
    ----------
    a: numpy.ndarray
        Array to decompose.
    axes: iter[int]
        Axes used to split `a`.
    sort: bool, optional
        If `True`, sort Schmidt decomposition.
    atol: float, optoinal
        Remove all Schmidt decomposition with weight smaller than `atol`.

    Returns
    -------
    s, uh, vh:
        Decomposition of `a` in `uh` and `vh`, with `uh` containing `axes`. `s`
        are the weights of the decomposition.

    See Also
    --------
    `scipy.linalg.svd`
    """
    from scipy.linalg import svd

    # Set defaults
    kwargs.setdefault('full_matrices', False)

    # Get array
    a = np.asarray(a)

    # Check axes
    axes = tuple(map(int, axes))

    # Check that there are no repeated axes
    if len(axes) != len(set(axes)):
        raise ValueError("Axes cannot be repeated in 'axes'.")

    # Check axes are within a dimenions
    if any(not 0 <= x < a.ndim for x in axes):
        raise ValueError("'axes' must be a list of valid 'a' axes.")

    # Get second half
    alt_axes = tuple(x for x in range(a.ndim) if x not in axes)

    # Get order
    order = axes + alt_axes

    # Save shape
    shape = a.shape

    # Get sizes of the left and right
    size_l = int(np.prod([shape[x] for x in axes]))
    size_r = int(np.prod([shape[x] for x in alt_axes]))

    # Check
    assert (size_l * size_r == np.prod(shape))

    # Transpose and reshape
    a = np.reshape(np.transpose(a, order), (size_l, size_r))

    # Apply SVD
    u, s, vh = svd(a, **kwargs)
    uh = u.T

    # Remove all weights below atol
    if atol:
        _sel = np.abs(s) >= atol
        s, uh, vh = s[_sel], uh[_sel], vh[_sel]

    # Sort if required
    if sort:
        _sel = np.argsort(s)
        s, uh, vh = s[_sel], uh[_sel], vh[_sel]

    # Check
    #assert (np.allclose(np.reshape(
    #    sum(s * np.kron(uh, vh) for s, uh, vh in zip(s, uh, vh)),
    #    (size_l, size_r)),
    #                    a,
    #                    atol=1e-5))

    # Reshape
    uh = np.reshape(uh, [len(s)] + [shape[x] for x in axes])
    vh = np.reshape(vh, [len(s)] + [shape[x] for x in alt_axes])

    return s, uh, vh


def isunitary(m: np.ndarray, atol: float = 1e-8) -> bool:
    """
    Check if `m` is a unitary matrix.

    Parameters
    ----------
    m: np.ndarray
        Matrix to check if unitary.
    atol: float, optional
        Absolute tollerance.

    Returns
    -------
    bool
        `True` if `m` is a unitary matrix, `False` otherwise
    """
    # Convert to np.ndarray
    m = np.asarray(m)

    # If m is not a square matrix, cannot be unitary
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False

    # Multiply with adjoint
    m1 = m.T.conj() @ m
    m2 = m @ m.T.conj()

    # Check if unitary
    return np.allclose(m1, m2, atol=atol) and np.allclose(
        m1, np.eye(m1.shape[0]), atol=atol)


def kron(a: np.ndarray, *cs: tuple[np.ndarray, ...], **kwargs):
    """
    Compute the Kronecker product among multiple arrays.

    Parameters
    ----------
    a, cs...: numpy.ndarray
        Arrays used to compute the Kronecker product

    Returns
    -------
    numpy.ndarray
        The Kronecker product.

    See Also
    --------
    numpy.kron
    """

    # Iteratively compute Kronecker product
    return kron(np.kron(a, cs[0]), *cs[1:]) if len(cs) else a


class globalize(object):
    """
    Globalize any function.
    """

    def __init__(self,
                 f: callable,
                 *,
                 name: str = None,
                 check_if_safe: bool = False):
        from copy import copy
        import sys

        # If name not provided ..
        if name is None:
            import uuid

            # Get initial node
            uuid.getnode()

            # Get random UUID
            name = uuid.uuid1()

            # Check if safely generated
            if check_if_safe and name.is_safe != uuid.SafeUUID.safe:
                raise RuntimeError(
                    f"Unique name '{name}' generated not in a safe way.")

        # Initialize
        self.__f = copy(f)
        self.__name = self.__f.__name__
        self.__qualname = self.__f.__qualname__
        self.__global_name = str(name)
        self.__namespace = sys.modules[self.__f.__module__]

    @property
    def f(self):
        return self.__f

    @property
    def name(self):
        return self.__global_name

    @property
    def namespace(self):
        return self.__namespace

    def __enter__(self):
        # Assign new name to function
        self.f.__name__ = self.f.__qualname__ = self.name

        # Add function to namespace
        setattr(self.namespace, self.name, self.f)

        # Return function
        return self.f

    def __exit__(self, type, value, traceback):
        # Return original name
        self.f.__name__ = self.__name
        self.f.__qualname__ = self.__qualname

        # Delete global name from namespace
        delattr(self.namespace, self.name)

    def __call__(self, *args, **kwargs):
        with self as f:
            return f(*args, **kwargs)


# Define new DeprecationWarning (to always print the warning signal)
class DeprecationWarning(Warning):
    pass


# Define new Warning (to always print the warning signal)
class Warning(Warning):
    pass


# Load library
def load_library(libname: str,
                 prefix: list[str, ...] = (None, 'lib', 'local/lib', 'usr/lib',
                                           'usr/local/lib')):
    from sys import base_prefix
    from os import path
    import ctypes

    # Define how to load library
    def _load(p: str = None):
        # Get library name
        _lib = libname if p is None else path.join(base_prefix, p, libname)

        # Try to load. If fails, return None
        try:
            return ctypes.cdll.LoadLibrary(_lib)
        except OSError:
            return None

    # Load library
    return next((lib for lib in map(_load, prefix) if lib is not None), None)
