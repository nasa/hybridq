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
from functools import lru_cache
import logging
import numpy as np
import autoray

from .compile import compile_lib
from .utils import get_lib_fn, load_library

__all__ = ['transpose']

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


@lru_cache
def get_swap_lib(nbytes: int, npos: int):
    """
    Return swap function.
    """

    # Compile the library
    compile_lib('swap', swap_npos=npos, swap_nbytes=nbytes)

    # Load and return function
    return get_lib_fn(load_library(f'hybridq_swap_{nbytes}_{npos}.so'), 'swap',
                      'int32', 'void*', 'uint32*', 'uint32')


# pylint: disable=undefined-variable
# pylint: disable=invalid-name
# pylint: disable=too-many-branches
def transpose(a: array_like,
              /,
              axes: array_like = None,
              *,
              inplace: bool = False,
              force_backend: bool = False,
              backend: str = None,
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
    from .aligned_array import asarray

    # Convert to numpy array
    a = asarray(a)

    # Convert axes to array
    axes = np.asarray(axes)

    # Axis must be a permutation of indexes
    if np.any(np.sort(axes) != np.arange(a.ndim)):
        raise ValueError("'axes' must be a valid permutation of indexes")

    # Get c-axes relative to the axes needed to be swapped
    c_axes = np.where(axes != np.arange(a.ndim))[0]
    if c_axes.size:
        c_axes = a.ndim - axes[c_axes[0]:][::-1] - 1

    # If no axes to swap are found, just return a
    else:
        return a

    # Check if hcore can be used
    _hcore_fails = []
    if len(c_axes) >= 20:
        _hcore_fails.append('Too many axes to swap')
    if a.shape != (2,) * a.ndim:
        _hcore_fails.append('Only binary axes are supported')
    if not a.flags.c_contiguous:
        _hcore_fails.append('Only c-contiguous arrays are allowed')

    # If no checks have failed ...
    if not force_backend and not _hcore_fails:

        # Swap
        try:
            # Copy if needed
            if not inplace:
                a = a.copy()

            # If 'a' is a complex type, view as float
            _complex = False
            if np.iscomplexobj(a):
                _complex = True
                a = a.view(a.real.dtype)
                c_axes = np.concatenate([[0], c_axes + 1])

            # Swap
            _err = get_swap_lib(a.dtype.itemsize,
                                len(c_axes))(a, c_axes, a.ndim + _complex)

            # Restore complex
            if _complex:
                a = a.view((1j * a.dtype.type([1])).dtype)

            # If there was an error, raise
            if _err:
                raise RuntimeError("Something went wrong")

            # Return swapped
            return a

        # If hcore fails ...
        # pylint: disable=broad-exception-caught
        except Exception as e:
            # Raise if required
            if raise_if_hcore_fails:
                raise e

            # Otherwise, log
            _LOGGER.warning(e)

    # Print reasons why hcore failed
    else:
        # Raise if required
        if not force_backend and raise_if_hcore_fails:
            raise RuntimeError('Cannot use HybridQ C++ core: ' +
                               ', '.join(_hcore_fails))

            # Otherwise, log
            _LOGGER.warning('Cannot use HybridQ C++ core: %s',
                            ', '.join(_hcore_fails))

    # Warn fallback
    _LOGGER.warning('Fallback to %s', 'backend' if backend is None else backend)

    # Return swapped
    return autoray.do('transpose', a, axes, like=backend)
