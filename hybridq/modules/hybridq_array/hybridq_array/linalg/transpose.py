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
from ..utils import load_library
import numpy as np

__all__ = ['transpose']

# Load library
_lib_swap = load_library('hybridq_swap.so')

# If library isn't loaded properly, warn
if _lib_swap is not None:
    from ..utils import define_lib_fn

    # Get _swap_core
    _swap_core = {}
    _swap_core.update({
        np.dtype(t + b): define_lib_fn(_lib_swap, f'swap_{t + b}', 'int',
                                       f'{t + b}*', 'uint32*', 'uint')
        for t in ('int', 'uint') for b in ('8', '16', '32', '64')
    })
    _swap_core.update({
        np.dtype(t + b): define_lib_fn(_lib_swap, f'swap_{t + b}', 'int',
                                       f'{t + b}*', 'uint32*', 'uint')
        for t in ('float',) for b in ('32', '64', '128')
    })
else:
    from warnings import warn
    warn("Cannot find C++ HybridQ core. "
         "Falling back to 'numpy.transpose' instead.")

    # Set core to none
    _swap_core = None


def transpose(a: array_like,
              axes: array_like = None,
              *,
              inplace: bool = False,
              force_backend: bool = False,
              backend: str = 'numpy',
              raise_if_hcore_fails: bool = False):
    """
    Transpose `a` accordingly to `axes`.

    Parameters
    ----------
    a: array_like
        Array to transpose.
    axes: array_like
        A permutation of axes of `a`.
    inplace: bool, optional
        If `True`, the transposition is performed inplace using the HybridQ C++
        core if possible. If `False`, same as `backend.transpose`.
    force_backend: bool, optional
        Force the use of `backend`, even if HybridQ C++ core can be used.
    backend: str, optional
        Backend to use in the case the HybridQ C++ core cannot be used.
    raise_if_hcore_fails: bool, optional
        If `True`, raise an error if the HybridQ C++ core cannot be used.
        Otherwise, a warning is emitted.

    Returns
    -------
    array_like
        The transposed array.

    See Also
    --------
    numpy.transpose
    """
    from ..aligned_array import array

    # Use backend if forced
    if force_backend:
        from autoray import do
        from warnings import warn
        warn(f"HybridQ C++ core is ignored. Fallback to '{backend}'")
        return do('transpose', a, axes, like=backend)

    # Convert a to array
    a = array(a, alignment=None, copy=not inplace)

    # Convert axes to array. If not provided, fully transpose array
    axes = np.arange(
        a.ndim).dtype('uint32')[::-1] if axes is None else np.asarray(
            axes, dtype='uint32')

    # Get position of first unordered axis
    n_ord = next((i for i, x in enumerate(axes) if i != x), len(axes))

    # If ordered, return
    if n_ord == len(axes):
        return a

    #  Check if hcore can be used
    _hcore_fail = []
    if _swap_core is None:
        _hcore_fail.append("Unable to load HybridQ C++ core")
    if a.dtype not in _swap_core:
        _hcore_fail.append(f"'{a.dtype}' not supported")
    if a.shape != (2,) * a.ndim:
        _hcore_fail.append("Only binary dimensions are allowed")
    if not a.flags.c_contiguous:
        _hcore_fail.append("Only 'C' order is supported")
    if not (3 < len(axes) - n_ord <= 16):
        _hcore_fail.append("'axes' order is not supported")

    # Warn if _hcore is not available
    if _hcore_fail:
        _hcore_fail = '; '.join(_hcore_fail)
        if raise_if_hcore_fails:
            raise AssertionError(f"Cannot use HybridQ C++ core: {_hcore_fail}")
        else:
            from warnings import warn
            warn(f"Cannot use HybridQ C++ core: {_hcore_fail}" +
                 f". Fallback to '{backend}'")

    # If hcore is available ...
    if not _hcore_fail:
        from ..utils import get_ctype

        # Get only unordered axes
        axes = axes[n_ord:]

        # Rescale axes
        axes = a.ndim - axes[::-1] - 1

        # Get pointer to a and axes
        _ptr_a = a.ctypes.data_as(get_ctype(str(a.dtype) + '*'))
        _ptr_axes = axes.ctypes.data_as(get_ctype(str(axes.dtype) + '*'))

        # Call library
        if _swap_core[a.dtype](_ptr_a, _ptr_axes, a.ndim, len(axes)):
            raise ValueError("Something went wrong.")

        # Return
        return a

    # Otherwise, just use numpy
    else:
        from autoray import do
        return np.transpose(a, axes)
        return do('transpose', a, axes, like=backend)
