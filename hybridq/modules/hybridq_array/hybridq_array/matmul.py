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
from string import ascii_letters
from functools import lru_cache
import logging

import numpy as np
import autoray

from .defaults import _DEFAULTS, Default, parse_default

__all__ = ['matmul']

# Create logger
_LOGGER = logging.getLogger(__name__)
_LOGGER.setLevel(logging.INFO)
_LOGGER_CH = logging.StreamHandler()
_LOGGER_CH.setLevel(logging.DEBUG)
_LOGGER_CH.setFormatter(
    logging.Formatter('%(asctime)s:%(name)s:%(levelname)s:%(message)s'))
_LOGGER.addHandler(_LOGGER_CH)


@lru_cache
def get_dot_lib(float_type: str,
                *,
                npos: int,
                max_log2_pack_size: int = None,
                tag: str = ''):
    """
    Return dot function.
    """

    from sysconfig import get_path
    from subprocess import Popen, PIPE
    import os

    from .compile import compile_lib, get_config
    from .utils import get_lib_fn, load_library

    # Get supported
    config_ = get_config()

    # Get optimal log2_pack_size
    if config_['avx512']:
        log2_pack_size = 5
    elif config_['avx2']:
        log2_pack_size = 5
    elif config_['avx']:
        log2_pack_size = 5
    else:
        log2_pack_size = 5

    # Get minimum between log2_pack_size and provided one
    if max_log2_pack_size is not None:
        log2_pack_size = min(log2_pack_size, max_log2_pack_size)

    try:
        # Compile the library
        compile_lib('dot',
                    dot_npos=npos,
                    dot_type=float_type,
                    dot_log2_pack_size=log2_pack_size)

        # Load and return function
        if tag[0] == 'q':
            return get_lib_fn(
                load_library(
                    f'hybridq_dot_{float_type}_{log2_pack_size}_{npos}.so'),
                'apply_' + tag, 'int64', float_type + '*', float_type + '*',
                float_type + '*', 'uint64*', 'uint64', 'uint64')
        else:
            return get_lib_fn(
                load_library(
                    f'hybridq_dot_{float_type}_{log2_pack_size}_{npos}.so'),
                'apply_' + tag, 'int64', float_type + '*', float_type + '*',
                'uint64*', 'uint64', 'uint64')
    except OSError as error:
        # Otherwise, log error ...
        _LOGGER.error(error)

        # ... and return None
        return None


@parse_default(_DEFAULTS, env_prefix='HYBRIDQ_ARRAY')
def matmul(a: matrix_like,
           b: array_like | tuple[array_like, array_like],
           /,
           axes: array_like,
           *,
           parallel: bool | int = Default,
           inplace: bool = False,
           force_backend: bool = False,
           backend: str = Default,
           raise_if_hcore_fails: bool = Default):
    """
    Multiply matrix `a` to array `b` at specific `axes`. For instance, for
    `a.ndim == 2`, this is equivalent to:

    numpy.einsum('ABab,...a...b...->...A...B...', a, b)

    `matmul` uses SIMD instructions whenever possible. Otherwise, `matmul`
    fallbacks to `backend`.

    Parameters
    ----------
    a, b: matrix_like, array_like | tuple[array_like, array_like]
        Matrix multiplication `a @ b` at specific `axes` for `b`. If `b` is a
        `tuple`, the first element of the tuple will be used as real part,
        while the second element as imaginary part.
    axes: array_like
        Axes for array `b` to which `a` is applied.
    parallel: bool | int, optional
        Run `transpose` using multi-threading. If `parallel == True`, all
        available threads are used. Otherwise, if `parallel` is
        int-convertible, a `parallel` number of threads is used instead.
    inplace: bool, optional
        If `True`, the multiplication is performed inplace whenever possible.
    force_backend: bool, optional
        Force the use of `backend`, even if HybridQ C++ core can be used.
    backend: str, optional
        Backend to use in the case the HybridQ C++ core cannot be used.
    raise_if_hcore_fails: bool, optional
        If `True`, raise an error if the HybridQ C++ core cannot be used.
        Otherwise, a warning is emitted.

    Returns
    -------
    a_@_b
        `a_@_b` being the result of the matrix/array multiplication.
    """
    # Check if 'b' is split into real and imaginary part
    is_b_tuple_ = isinstance(b, tuple) and len(b) == 2

    # Convert to numpy arrays
    a = np.asarray(a)
    b = tuple(map(np.asarray, b)) if is_b_tuple_ else np.asarray(b)

    # 'a' must be a square matrix
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("'a' must be a square matrix")

    # If 'b' is tuple, check consistency
    if is_b_tuple_:
        if b[0].shape != b[1].shape or b[0].dtype != b[
                1].dtype or np.iscomplexobj(b[0]):
            raise ValueError("'b' is not valid")

    # Convert axes
    axes = np.arange(a.shape[1]) if axes is None else np.asarray(axes)

    # Check axes
    if axes.ndim != 1 or a.shape[1] != np.prod(
            np.asarray((b[0] if is_b_tuple_ else b).shape)[axes]) or np.any(
                axes >= (b[0] if is_b_tuple_ else b).ndim) or len(
                    set(axes)) != axes.size:
        raise ValueError("'axes' is not valid")

    # Check if a/b are complex
    complex_a_ = np.iscomplexobj(a)
    complex_b_ = True if is_b_tuple_ else np.iscomplexobj(b)

    # Check if hcore can be used
    _hcore_fails = []
    if any(b.shape != (2,) * b.ndim for b in (b if is_b_tuple_ else [b])):
        _hcore_fails.append("Only binary dimensions are supported for 'b'")
    if axes.size > 18:
        _hcore_fails.append("Too many axes")
    if not a.flags.c_contiguous:
        _hcore_fails.append("'a' must be c-contiguous")
    if any(not b.flags.c_contiguous for b in (b if is_b_tuple_ else [b])):
        _hcore_fails.append("'b' must be c-contiguous")

    # If no checks have failed ...
    if not force_backend and not _hcore_fails:
        # Get common type
        dtype_ = (a.dtype.type([1]) +
                  (b[0].dtype.type([1]) + 1j * b[1].dtype.type([1])
                   if is_b_tuple_ else b.dtype.type([1]))).dtype

        # Check if complex
        complex_ = np.iscomplexobj(dtype_.type([1]))

        # Get real type
        rtype_ = dtype_.type([1]).real.dtype if complex_ else dtype_

        # Get positions
        pos_ = (b[0] if is_b_tuple_ else b).ndim - np.asarray(
            axes[::-1], dtype='uint64') - 1

        # Get maximum allower size of pack
        max_log2_pack_size_ = np.sort(pos_)[0]

        # Get right tag
        if not complex_a_ and not complex_b_:
            tag_ = 'rr'
        else:
            tag_ = ('q' if is_b_tuple_ else 'c') + ('c' if complex_a_ else 'r')

        # Load library
        lib_ = get_dot_lib(str(rtype_),
                           npos=axes.size,
                           max_log2_pack_size=max_log2_pack_size_,
                           tag=tag_)

        if lib_ is not None:
            # Parallel?
            parallel_ = int(not parallel) if isinstance(parallel,
                                                        bool) else int(parallel)

            # Convert both a and b to the common type
            a_ = np.asarray(a, dtype=dtype_ if complex_a_ else rtype_)
            if is_b_tuple_:
                b_ = list(map(lambda x: np.asarray(x, dtype=rtype_), b))
                if not inplace:
                    if b[0].ctypes.data == b_[0].ctypes.data:
                        b_[0] = np.copy(b_[0])
                    if b[1].ctypes.data == b_[1].ctypes.data:
                        b_[1] = np.copy(b_[1])
            else:
                b_ = np.asarray(b, dtype=dtype_)
                if not inplace and b.ctypes.data == b_.ctypes.data:
                    b_ = np.copy(b_)

            # Call library
            if is_b_tuple_:
                if not lib_(b_[0].view(rtype_), b_[1].view(rtype_),
                            a_.view(rtype_), pos_, b[0].ndim, parallel_):
                    return b_
            else:
                if not lib_(b_.view(rtype_), a_.view(rtype_), pos_, b.ndim,
                            parallel_):
                    return b_

        # If it didn't return, there was an error in running the library
        _hcore_fails.append("Cannot use C++ library")

    # Raise if needed
    if _hcore_fails:
        if force_backend or not raise_if_hcore_fails:
            # Log if backend was forced
            _LOGGER.warning('Cannot use HybridQ C++ core: %s',
                            ', '.join(_hcore_fails))
        else:
            # Otherwise, raise
            raise RuntimeError('Cannot use HybridQ C++ core: ' +
                               ', '.join(_hcore_fails))

    # Warn fallback
    _LOGGER.warning('Fallback to %s', 'backend' if backend is None else backend)

    # If tuple, convert to to array
    if is_b_tuple_:
        b = b[0] + 1j * b[1]

    # Build map
    _map = ''.join(
        ascii_letters[i + axes.size] for i in range(axes.size)) + ''.join(
            ascii_letters[i] for i in range(axes.size))
    _map += ','
    _map += ''.join(ascii_letters[axes.tolist().index(i)] if i in
                    axes else ascii_letters[i + 2 * axes.size]
                    for i in range(b.ndim))
    _map += '->'
    _map += ''.join(
        ascii_letters[axes.tolist().index(i) +
                      axes.size] if i in axes else ascii_letters[i +
                                                                 2 * axes.size]
        for i in range(b.ndim))

    # Reshape a to the right shape
    a = np.reshape(a, np.concatenate([np.array(b.shape)[axes]] * 2))

    # Return
    return autoray.do('einsum', _map, a, b, like=backend)
