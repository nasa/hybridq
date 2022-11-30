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
from .array import Array
import numpy as np

HANDLED_FUNCTIONS = {}


def implements(numpy_function):

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


###############################################################################


@implements(np.iscomplexobj)
def iscomplexobj(x):
    return x.iscomplexobj()


###############################################################################


@implements(np.may_share_memory)
def may_share_memory(a, b):
    # Short
    _fn = np.may_share_memory

    if isinstance(a, Array) and isinstance(b, Array):
        return any(
            _fn(x, y)
            for x in [a._real, a._imag]
            for y in [b._real, b._imag]
            if x is not None and y is not None)
    elif isinstance(a, Array):
        return any(_fn(x, b) for x in [a._real, a._imag] if x is not None)
    elif isinstance(b, Array):
        return any(_fn(a, y) for y in [b._real, b._imag] if y is not None)
    else:
        return _fn(a, b)


@implements(np.shares_memory)
def shares_memory(a, b):
    # Short
    _fn = np.shares_memory

    if isinstance(a, Array) and isinstance(b, Array):
        return any(
            _fn(x, y)
            for x in [a._real, a._imag]
            for y in [b._real, b._imag]
            if x is not None and y is not None)
    elif isinstance(a, Array):
        return any(_fn(x, b) for x in [a._real, a._imag] if x is not None)
    elif isinstance(b, Array):
        return any(_fn(a, y) for y in [b._real, b._imag] if y is not None)
    else:
        return _fn(a, b)


###############################################################################


def _like(a, fn_name_r, fn_name_i):
    from .aligned_array import empty, zeros, ones

    # Get function
    _fn_r = {'empty': empty, 'ones': ones, 'zeros': zeros}[fn_name_r]
    _fn_i = {'empty': empty, 'ones': ones, 'zeros': zeros}[fn_name_i]

    # Get rtype
    _rtype = a.dtype.type(1).real.dtype

    # Get buffers
    _r = _fn_r(shape=a.shape,
               dtype=_rtype,
               alignment=a.min_alignment,
               order='C' if a.real.flags.c_contiguous else 'F')
    _i = _fn_i(shape=a.shape,
               dtype=_rtype,
               alignment=a.min_alignment,
               order='C' if a.real.flags.c_contiguous else
               'F') if a.iscomplexobj() else None

    # Return
    return Array(buffer=(_r, _i, a.min_alignment))


@implements(np.empty_like)
def empty_like(a):
    return _like(a, 'empty', 'empty')


@implements(np.zeros_like)
def zeros_like(a):
    return _like(a, 'zeros', 'zeros')


@implements(np.ones_like)
def ones_like(a):
    return _like(a, 'ones', 'zeros')


###############################################################################


@implements(np.allclose)
def allclose(a, b, *args, **kwargs):
    # If both are Array's ...
    if isinstance(a, Array) and isinstance(b, Array):
        # Check real parts
        if not np.allclose(a.real, b.real, *args, **kwargs):
            return False

        # Check imaginary parts
        if a.iscomplexobj() and b.iscomplexobj():
            return np.allclose(a.imag, b.imag, *args, **kwargs)

        elif a.iscomplexobj():
            return np.allclose(a.imag, 0, *args, **kwargs)

        elif b.iscomplexobj():
            return np.allclose(b.imag, 0, *args, **kwargs)

        else:
            return True

    # ... otherwise, convert
    else:
        return np.allclose(np.asarray(a), np.asarray(b), *args, **kwargs)


###############################################################################


@implements(np.sort)
def sort(a, *args, **kwargs):
    from .aligned_array import asarray

    # Get order
    _order = 'C' if a.real.flags.c_contiguous else 'F'

    if a.iscomplexobj():
        _b = np.sort(np.asarray(a), *args, **kwargs)
        _r = asarray(_b.real, alignment=a.min_alignment, order=_order)
        _i = asarray(_b.imag, alignment=a.min_alignment, order=_order)
    else:
        _r = asarray(np.sort(a.real, *args, **kwargs),
                     alignment=a.min_alignment,
                     order=_order)
        _i = None

    # Return
    return Array(buffer=(_r, _i, a.min_alignment))


###############################################################################


def __unary_func__(a, func, *args, **kwargs):
    from .aligned_array import asarray
    _r = asarray(func(a.real, *args, **kwargs), alignment=a.min_alignment)
    _i = asarray(func(a.imag, *args, **kwargs),
                 alignment=a.min_alignment) if np.iscomplexobj(a) else None
    return Array(buffer=(_r, _i, a.min_alignment))


@implements(np.reshape)
def reshape(a, *args, **kwargs):
    return __unary_func__(a, np.reshape, *args, **kwargs)


@implements(np.transpose)
def transpose(a, *args, **kwargs):
    from .linalg import transpose
    return __unary_func__(a, transpose, *args, **kwargs)


###############################################################################


@implements(np.linalg.norm)
def norm(a, *args, **kwargs):
    # Get real part
    _norm = np.linalg.norm(a.real, *args, **kwargs)

    # Get imaginary part
    if a.iscomplexobj():
        _norm = np.sqrt(_norm**2 + np.linalg.norm(a.imag, *args, **kwargs)**2)

    # Return
    return _norm


###############################################################################


@implements(np.tensordot)
def tensordot(a, b, *args, **kwargs):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.tensordot(a, b, *args, **kwargs)


@implements(np.einsum)
def einsum(a, b, *args, **kwargs):
    a = np.asarray(a)
    b = np.asarray(b)
    return np.einsum(a, b, *args, **kwargs)
