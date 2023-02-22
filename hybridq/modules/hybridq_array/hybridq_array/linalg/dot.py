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
from ..defaults import _DEFAULTS
from ..utils import load_library
import numpy as np

__all__ = ['tensordot']

# Load library
_lib_dot = load_library('hybridq.so')

# If library isn't loaded properly, warn
if _lib_dot is not None:
    from ..utils import define_lib_fn

    # Get _dot_core
    _dot_core = {
        np.dtype(t + b):
        define_lib_fn(_lib_dot, f'apply_U_{t + b}', 'int', f'{t + b}*',
                      f'{t + b}*', f'{t + b}*', 'uint32*', 'uint', 'uint')
        for t in ('float',) for b in ('32', '64', '128')
    }

    # Get size of pack
    _log2_pack_size = define_lib_fn(_lib_dot, "get_log2_pack_size", 'uint32')()
else:
    from warnings import warn
    warn("Cannot find C++ HybridQ core. "
         "Falling back to 'numpy.transpose' instead.")

    # Set core to None
    _dot_core = None

    # Set size of pack to None
    _log2_pack_size = None


def _get_swap_order(axes, ndim):
    # Get axes within the pack
    _axes = axes[axes < _log2_pack_size]

    # If no swap is needed, just return
    if len(_axes) == 0:
        return np.asarray(axes, dtype='uint32'), np.array([])

    # Get free positions to swap to
    _swap = np.setdiff1d(np.arange(_log2_pack_size, ndim), axes)[:len(_axes)]

    # Initialize order
    _order = np.arange(_swap[-1] + 1).astype('uint32')

    # Swap axes
    _order[_axes], _order[_swap] = _order[_swap], _order[_axes]

    # Get new axes
    _axes = np.array(
        [np.where(_order == y)[0][0] if y < len(_order) else y for y in axes],
        dtype='uint32')

    # Return order
    return _axes, _order


def _tensordot_contract(a: array_like, b: array_like, /, axes, *, backend: str):
    from opt_einsum import get_symbol, contract
    from autoray import do

    # Build path
    _path = ''
    _path += ''.join(get_symbol(i + b.ndim) for i in axes)
    _path += ''.join(get_symbol(i) for i in axes)
    _path += ','
    _path += ''.join(get_symbol(i) for i in range(b.ndim))
    _path += '->'
    _path += ''.join(
        get_symbol(i + b.ndim if i in axes else i) for i in range(b.ndim))

    # Convert to array
    a = do('asarray', a, like=backend)
    b = do('asarray', b, like=backend)

    # Contract
    return contract(_path, a, b, backend=backend)


def tensordot(a: array_like,
              b: array_like,
              /,
              axes,
              *,
              swap_back: bool = True,
              inplace: bool = False,
              force_backend: bool = False,
              backend: str = _DEFAULTS['backend'],
              raise_if_hcore_fails: bool = _DEFAULTS['raise_if_hcore_fails']):
    from ..utils import get_ctype
    from ..array import array
    """
    Apply tensor multiplication between `a` and `b` between common `axes`. For
    instance, for `a.ndim == 2`, this is equivalent to:

    numpy.einsum('ABab,...a...b->...A...B...', a, b)

    `tensordot` uses AVX instructions that may require to move axes from the
    least significative positions to higher positions. If `swap_back == True`,
    the moved axes are brought back to the original positions with the
    additional cost of a transposition.  Otherwise, if `swap_back == False`, an
    `order` is returned such that:

    a = tensordot(..., swap_back=True)
    b, order = tensordot(..., swap_back=False)

    assert(
      numpy.allclose(
        a,
        numpy.transpose(b, order)
      )
    )

    Parameters
    ----------
    a, b: array_like
        Arrays to multiply using tensor contraction. All axes from `a` are
        used (in order), while only axes provided in `axes` are used for `b`.
    axes: array_like
        Axes to use for `b`.
    swap_back: bool, optional
        `tensordot` uses AVX instructions that may require to move axes from
        the least significative positions to higher positions. If `True`, the
        moved axes are brought back to the original positions with the
        additional cost of a transposition.  Otherwise, if `False`, an `order`
        is returned
    inplace: bool, optional
        If `True`, the multiplication is performed inplace using the HybridQ
        C++ core if possible. If `False`, same as `backend.transpose`. Only `b`
        is affected.
    force_backend: bool, optional
        Force the use of `backend`, even if HybridQ C++ core can be used.
    backend: str, optional
        Backend to use in the case the HybridQ C++ core cannot be used.
    raise_if_hcore_fails: bool, optional
        If `True`, raise an error if the HybridQ C++ core cannot be used.
        Otherwise, a warning is emitted.

    Returns
    -------
    a_tensordot_b[, order]
        `a_tensordot_b` being the result of the tensor multiplication. If
        `swap_back == False`, also return `order`.
    """
    # Use backend if forced
    if force_backend:
        from warnings import warn
        warn(f"HybridQ C++ core is ignored. Fallback to '{backend}'")

        # Get contraction
        _c = _tensordot_contract(a, b, axes=axes, backend=backend)

        # Return
        return _c if swap_back else (_c, np.arange(_c.ndim))

    # Convert to array
    axes = np.asarray(axes)
    b = array(b, order='C', copy=not inplace)
    a = np.asarray(a, order='C', dtype=b.dtype)

    # Get real type
    _rtype = b.dtype.type(1).real.dtype

    #  Check if hcore can be used
    _hcore_fail = []
    if _dot_core is None:
        _hcore_fail.append("Unable to load HybridQ C++ core")
    if not b.iscomplexobj():
        _hcore_fail.append("'b' must be an array of complex numbers")
    if a.shape != (2,) * a.ndim or a.ndim != 2 * len(axes):
        _hcore_fail.append("'a' is not consistent with 'axes'")
    if _rtype not in _dot_core:
        _hcore_fail.append(f"'{b.dtype}' not supported")
    if b.shape != (2,) * b.ndim:
        _hcore_fail.append("Only binary dimensions are allowed")
    if b.ndim < 6:
        _hcore_fail.append("'b' is too small")
    if axes.ndim != 1 or np.any(axes >= b.ndim):
        _hcore_fail.append("'axes' must be a list of valid axes")
    if (b.ndim - axes.size) < _log2_pack_size or axes.size > 10:
        _hcore_fail.append("Too many axes")

    # Rescale axes
    _axes = (b.ndim - axes[::-1] - 1)

    # Get swap order
    _axes, _swap_order = _get_swap_order(_axes, b.ndim)

    # Check swap_size
    if len(_swap_order) > 12:
        _hcore_fail.append("Too many axes to swap")

    # Warn if _hcore is not available
    if _hcore_fail:
        _hcore_fail = '; '.join(_hcore_fail)
        if raise_if_hcore_fails:
            raise AssertionError(f"Cannot use HybridQ C++ core: {_hcore_fail}")
        else:
            from warnings import warn
            warn(f"Cannot use HybridQ C++ core: {_hcore_fail}" +
                 f". Fallback to '{backend}'")

    # Use HybridQ C++ core if possible
    if not _hcore_fail:
        # Get pointers
        _a_ptr = a.ctypes.data_as(get_ctype(str(_rtype) + '*'))
        _b_re_ptr = b.real.ctypes.data_as(get_ctype(str(_rtype) + '*'))
        _b_im_ptr = b.imag.ctypes.data_as(get_ctype(str(_rtype) + '*'))
        _swap_order_ptr = _swap_order.ctypes.data_as(get_ctype('uint32*'))
        _axes_ptr = _axes.ctypes.data_as(get_ctype('uint32*'))

        # Swap if needed
        if len(_swap_order):
            from .transpose import get_swap_lib
            get_swap_lib(b.real.dtype, len(_swap_order))(b.real, _swap_order,
                                                         b.ndim)
            get_swap_lib(b.imag.dtype, len(_swap_order))(b.imag, _swap_order,
                                                         b.ndim)

        # Apply matrix
        if _dot_core[_rtype](_b_re_ptr, _b_im_ptr, _a_ptr, _axes_ptr, b.ndim,
                             len(_axes)):
            raise RuntimeError("Something went wrong")

        # Swap back if needed
        if len(_swap_order) and swap_back:
            from .transpose import get_swap_lib
            get_swap_lib(b.real.dtype, len(_swap_order))(b.real, _swap_order,
                                                         b.ndim)
            get_swap_lib(b.imag.dtype, len(_swap_order))(b.imag, _swap_order,
                                                         b.ndim)

        # Otherwise, build final order
        else:
            _swap_order = b.ndim - np.concatenate(
                (_swap_order, np.arange(len(_swap_order), b.ndim)))[::-1] - 1

        # Return
        return b if swap_back else (b, _swap_order)

    else:
        # Get contraction
        _c = _tensordot_contract(a, b, axes=axes, backend=backend)

        # Return
        return _c if swap_back else (_c, np.arange(_c.ndim))
