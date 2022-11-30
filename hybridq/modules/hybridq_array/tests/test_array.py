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
from hybridq_array.aligned_array import get_alignment
from functools import partial as partial_fn
from hybridq_array.utils import load_library
from hybridq_array.array import *
import numpy as np
import pytest

# Define assert_allclose
assert_allclose = partial_fn(np.testing.assert_allclose, atol=1e-5)


@pytest.fixture(autouse=True)
def set_seed():
    from numpy import random
    from os import environ
    from sys import stderr

    # Get random seed
    seed = random.randint(2**32 - 1)

    # Get state
    state = random.get_state()

    # Set seed
    random.seed(seed)

    # Print seed
    print(f"# Used seed [{environ['PYTEST_CURRENT_TEST']}]: {seed}",
          file=stderr)

    # Wait for PyTest
    yield

    # Set state
    random.set_state(state)


@pytest.mark.parametrize(
    'dtype,order,alignment',
    [(dtype, order, alignment)
     for dtype in ['complex64', 'complex128', 'complex256'] for order in 'CF'
     for alignment in [32, 64, 128, 256, 512, 1024]])
def test__Array_complex(dtype, order, alignment):
    from hybridq_array.aligned_array import asarray

    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(dtype)
    imag = (100 * np.random.random(size=shape)).astype(dtype)

    # Get new array
    array = Array(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Check
    assert (array.shape == shape)
    assert (array.dtype == dtype)
    assert (array.real.shape == shape)
    assert (array.real.dtype == rtype)
    assert (array.imag.shape == shape)
    assert (array.imag.dtype == rtype)
    assert ((order == 'C' and array.real.flags.c_contiguous) or
            (order == 'F' and array.real.flags.f_contiguous))
    assert ((order == 'C' and array.imag.flags.c_contiguous) or
            (order == 'F' and array.imag.flags.f_contiguous))
    assert (array.min_alignment == alignment)

    # Assign
    array[:] = real + 1j * imag

    # Check
    assert_allclose(array.real, real)
    assert_allclose(array.imag, imag)
    assert (isinstance(array.__array__(), np.ndarray))
    assert (isinstance(array.tolist(), list))
    assert_allclose(np.asarray(array), real + 1j * imag)

    # Assign
    array[:] = real

    # Check
    assert_allclose(array.real, real)
    assert_allclose(array.imag, 0)

    # Assign
    array[:] = 1j * imag

    # Check
    assert_allclose(array.real, 0)
    assert_allclose(array.imag, imag)

    # Assign
    array.real[:] = real
    array.imag[:] = imag

    # Check
    assert_allclose(array.real, real)
    assert_allclose(array.imag, imag)

    # Assigning non AlignedArray should fail
    try:
        array = Array(buffer=(real, imag, alignment))
    except TypeError as e:
        assert (str(e) == f"'{type(real).__name__}' is not supported")

    # Convert to AlignedArray
    real = asarray(real, dtype=rtype, order=order, alignment=alignment)
    imag = asarray(imag, dtype=rtype, order=order, alignment=alignment)

    # Get array
    array = Array(buffer=(real, imag, alignment))

    # Check
    assert (array.shape == shape)
    assert (array.dtype == dtype)
    assert ((order == 'C' and array.real.flags.c_contiguous) or
            (order == 'F' and array.real.flags.f_contiguous))
    assert ((order == 'C' and array.imag.flags.c_contiguous) or
            (order == 'F' and array.imag.flags.f_contiguous))
    assert (array.min_alignment == alignment)
    assert (np.may_share_memory(array.real, real))
    assert (np.may_share_memory(array.imag, imag))

    # Create array
    array = Array(shape=shape, dtype=rtype, order=order, alignment=alignment)

    # If Array is real, cannot assign imaginary part
    try:
        array[:] = real + 1j * imag
    except NotImplementedError:
        pass


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__Array_1(dtype, order, alignment):
    from hybridq_array.aligned_array import asarray

    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(rtype)
    imag = None if dtype == rtype else (
        100 * np.random.random(size=shape)).astype(rtype)

    # Get new array
    array = Array(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Assign
    array[:] = real if dtype == rtype else real + 1j * imag

    # Check
    assert (array.shape == shape)
    assert (array.dtype == dtype)
    assert (array.real.shape == shape)
    assert (array.real.dtype == rtype)
    assert ((order == 'C' and array.real.flags.c_contiguous) or
            (order == 'F' and array.real.flags.f_contiguous))
    assert (array.min_alignment == alignment)
    if array.iscomplexobj():
        assert (array.imag.shape == shape)
        assert (array.imag.dtype == rtype)
        assert ((order == 'C' and array.imag.flags.c_contiguous) or
                (order == 'F' and array.imag.flags.f_contiguous))
    assert (isinstance(array.__array__(), np.ndarray))
    assert (isinstance(array.tolist(), list))
    assert (len(array) == array.shape[0])
    assert (array.size == np.prod(array.shape))

    # Get random items
    _p = [tuple(map(np.random.randint, array.shape)) for _ in range(10_000)]
    _a = np.array(list(map(lambda x: array[x], _p)))
    _r = np.array(list(map(lambda x: real[x], _p)))
    _i = np.zeros_like(_r) if imag is None else np.array(
        list(map(lambda x: imag[x], _p)))
    assert_allclose(_a, _r + 1j * _i)

    # Get random items
    _p = [tuple(map(np.random.randint, array.shape[3:])) for _ in range(10_000)]
    _q = tuple(map(np.random.randint, array.shape[:3]))
    _a = np.array(list(map(lambda x: array[_q][x], _p)))
    _r = np.array(list(map(lambda x: real[_q][x], _p)))
    _i = np.zeros_like(_r) if imag is None else np.array(
        list(map(lambda x: imag[_q][x], _p)))
    assert_allclose(_a, _r + 1j * _i)

    # Set random items
    _p = tuple((np.random.randint if p < 0.5 else slice)(s)
               for s, p in zip(shape, np.random.random(len(shape))))
    _r = (100 * np.random.random(size=real[_p].shape)).astype(rtype)
    _i = (100 * np.random.random(size=real[_p].shape)).astype(rtype)
    real[_p] = _r
    if imag is not None:
        imag[_p] = _i
    array[_p] = _r if imag is None else _r + 1j * _i
    assert_allclose(array[_p].real, _r)
    if imag is not None:
        assert_allclose(array[_p].imag, _i)
    assert_allclose(array, real if imag is None else real + 1j * imag)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__Array_2(dtype, order, alignment):
    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(rtype)
    imag = None if dtype == rtype else (
        100 * np.random.random(size=shape)).astype(rtype)
    _array = real if imag is None else real + 1j * imag

    # Get new array
    array = Array(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Assign
    array[:] = _array

    # Check min_alignment
    assert (array.min_alignment == alignment)
    assert (array.alignment == min(array.real.alignment, array.imag.alignment)
            if array.iscomplexobj() else array.real.alignment)
    assert (array.dtype == dtype)
    assert (array.shape == shape)
    assert (len(array) == shape[0])
    assert (array.size == np.prod(shape))
    assert (array.ndim == len(shape))
    assert (array.iscomplexobj() == ('complex' in str(dtype)))

    # Check align (default, copy == True)
    _max_alignment = max(array.real.alignment, array.imag.alignment)
    _array = array.align(array.alignment // 2)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == array.alignment // 2)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)

    # Check align (copy == False)
    _alignment = array.alignment // 2
    _array = array.align(_alignment, copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == _alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _alignment = array.alignment
    _array = array.align(_alignment, copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == _alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _alignment = 2 * max(array.real.alignment, array.imag.alignment)
    _array = array.align(_alignment, copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == _alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)

    # Check copy
    _array = array.copy()
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)

    # Check ravel
    _array = array.ravel()
    assert (isinstance(_array, Array))
    assert (_array.shape == (array.size,))
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (np.may_share_memory(array.real, _array.real) == (order == 'C'))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag) == (order == 'C'))
    assert (np.may_share_memory(array, _array) == (order == 'C'))
    assert_allclose(_array.ravel(),
                    array.real.ravel() + 1j * array.imag.ravel())

    # Check flatten
    _array = array.flatten()
    assert (isinstance(_array, Array))
    assert (_array.shape == (array.size,))
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array.ravel(),
                    array.real.ravel() + 1j * array.imag.ravel())

    # Check conj
    _array = array.conj()
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real - 1j * array.imag)

    # Check T
    _array = array.T
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape[::-1])
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real.T + 1j * array.imag.T)

    # Check astype (default, copy == True)
    _max_alignment = max(array.real.alignment, array.imag.alignment)
    _array = array.astype(dtype=dtype, order=order, alignment=array.alignment)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == array.alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _array = array.astype(dtype=dtype,
                          order=order,
                          alignment=array.alignment // 2)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == array.alignment // 2)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _array = array.astype(dtype=dtype,
                          order=order,
                          alignment=2 * _max_alignment)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == 2 * _max_alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _dtype = 'float' if 'complex' in str(dtype) else 'complex'
    _order = 'C' if order == 'F' else 'F'
    _array = array.astype(dtype=_dtype, order=_order, alignment=alignment)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert ((_order == 'C' and _array.real.flags.c_contiguous) or
            (_order == 'F' and _array.real.flags.f_contiguous))
    assert ((_order == 'C' and _array.imag.flags.c_contiguous) or
            (_order == 'F' and _array.imag.flags.f_contiguous))
    assert (_array.dtype == _dtype)
    assert (_array.min_alignment == alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real.astype(_dtype))
    if _dtype == 'complex':
        assert_allclose(_array.imag, 0)
    else:
        assert (_array._imag == None)

    # Check astype (copy == False)
    _max_alignment = max(array.real.alignment, array.imag.alignment)
    _alignment = array.alignment
    _array = array.astype(dtype=dtype,
                          order=order,
                          alignment=_alignment,
                          copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == _alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _alignment = array.alignment // 2
    _array = array.astype(dtype=dtype,
                          order=order,
                          alignment=_alignment,
                          copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == _alignment)
    assert (np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (np.may_share_memory(array.imag, _array.imag))
    assert (np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _array = array.astype(dtype=dtype,
                          order=order,
                          alignment=2 * _max_alignment,
                          copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert (_array.dtype == dtype)
    assert (_array.min_alignment == 2 * _max_alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real + 1j * array.imag)
    #
    _dtype = 'float' if 'complex' in str(dtype) else 'complex'
    _order = 'C' if order == 'F' else 'F'
    _array = array.astype(dtype=_dtype,
                          order=_order,
                          alignment=alignment,
                          copy=False)
    assert (isinstance(_array, Array))
    assert (_array.shape == shape)
    assert ((_order == 'C' and _array.real.flags.c_contiguous) or
            (_order == 'F' and _array.real.flags.f_contiguous))
    assert ((_order == 'C' and _array.imag.flags.c_contiguous) or
            (_order == 'F' and _array.imag.flags.f_contiguous))
    assert (_array.dtype == _dtype)
    assert (_array.min_alignment == alignment)
    assert (not np.may_share_memory(array.real, _array.real))
    if array.iscomplexobj():
        assert (not np.may_share_memory(array.imag, _array.imag))
    assert (not np.may_share_memory(array, _array))
    assert_allclose(_array, array.real.astype(_dtype))
    if _dtype == 'complex':
        assert_allclose(_array.imag, 0)
    else:
        assert (_array._imag == None)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__Array_np(dtype, order, alignment):
    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(rtype)
    imag = None if dtype == rtype else (
        100 * np.random.random(size=shape)).astype(rtype)
    _array = real if imag is None else real + 1j * imag

    # Get new array
    array = Array(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Assign
    array[:] = _array

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(rtype)
    imag = None if dtype == rtype else (
        100 * np.random.random(size=shape)).astype(rtype)
    _array = real if imag is None else real + 1j * imag

    # Check allclose
    assert_allclose(array, array)
    assert_allclose(array, np.asarray(array))
    try:
        assert_allclose(array, _array)
    except AssertionError:
        pass
    else:
        raise AssertionError

    # Check norm
    assert_allclose(np.linalg.norm(array), np.linalg.norm(np.asarray(array)))

    # Check empty
    _array = np.empty_like(array)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (_array.real.flags.c_contiguous == array.real.flags.c_contiguous)
    if array.iscomplexobj():
        assert (_array.imag.flags.c_contiguous == array.imag.flags.c_contiguous)
    else:
        assert (_array._imag is None)
    assert (_array.min_alignment == array.min_alignment)
    assert (not np.may_share_memory(_array, array))

    # Check zeros
    _array = np.zeros_like(array)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (_array.real.flags.c_contiguous == array.real.flags.c_contiguous)
    if array.iscomplexobj():
        assert (_array.imag.flags.c_contiguous == array.imag.flags.c_contiguous)
    else:
        assert (_array._imag is None)
    assert (_array.min_alignment == array.min_alignment)
    assert (not np.may_share_memory(_array, array))
    assert_allclose(_array, 0)

    # Check ones
    _array = np.ones_like(array)
    assert (isinstance(_array, Array))
    assert (_array.shape == array.shape)
    assert (_array.dtype == array.dtype)
    assert (_array.min_alignment == alignment)
    assert (_array.real.flags.c_contiguous == array.real.flags.c_contiguous)
    if array.iscomplexobj():
        assert (_array.imag.flags.c_contiguous == array.imag.flags.c_contiguous)
    else:
        assert (_array._imag is None)
    assert (_array.min_alignment == array.min_alignment)
    assert (not np.may_share_memory(_array, array))
    assert_allclose(_array, 1)

    # Check reshape
    _shape = np.random.permutation(shape)
    _array_1 = np.reshape(array, _shape)
    _array_2 = np.reshape(np.asarray(array), _shape)
    assert (isinstance(_array_1, Array))
    assert (_array_1.shape == _array_2.shape)
    assert (_array_1.dtype == _array_2.dtype)
    assert (_array.min_alignment == alignment)
    assert (_array.min_alignment == array.min_alignment)
    assert (not np.may_share_memory(_array, array))
    assert_allclose(_array_1, _array_2)

    # Check sort
    _axis = np.random.randint(array.ndim)
    _array_1 = np.sort(array, axis=_axis)
    _array_2 = np.sort(np.asarray(array), axis=_axis)
    assert (isinstance(_array_1, Array))
    assert (_array_1.shape == _array_2.shape)
    assert (_array_1.dtype == _array_2.dtype)
    assert (_array.min_alignment == alignment)
    assert (_array.min_alignment == array.min_alignment)
    assert (not np.may_share_memory(_array, array))
    assert_allclose(_array_1, _array_2)

    # Check tranpose
    if not load_library('hybridq.so') or not load_library('hybridq_swap.so'):
        pytest.skip("Cannot load HybridQ C++ core")
    else:
        _shape = np.random.permutation(array.ndim)
        _array_1 = np.transpose(array, _shape)
        _array_2 = np.transpose(np.asarray(array), _shape)
        assert (isinstance(_array_1, Array))
        assert (_array_1.shape == _array_2.shape)
        assert (_array_1.dtype == _array_2.dtype)
        assert (_array.min_alignment == alignment)
        assert (_array.min_alignment == array.min_alignment)
        assert (not np.may_share_memory(_array, array))
        assert_allclose(_array_1, _array_2)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__array_asarray(dtype, order, alignment):
    from hybridq_array.aligned_array import get_alignment
    from hybridq_array.aligned_array import empty
    from hybridq_array import _DEFAULTS

    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random real and imag
    real = (100 * np.random.random(size=shape)).astype(rtype, order=order)
    imag = None if dtype == rtype else (100 *
                                        np.random.random(size=shape)).astype(
                                            rtype, order=order)
    _array = real if imag is None else real + 1j * imag

    # Get new array
    array_1 = asarray(_array)

    # Checks
    assert_allclose(array_1, _array)
    assert (isinstance(array_1, Array))
    assert (array_1.dtype == _array.dtype)
    assert ((array_1.real.flags.c_contiguous and order == 'C') or
            (array_1.real.flags.f_contiguous and order == 'F'))
    if array_1.iscomplexobj():
        assert ((array_1.imag.flags.c_contiguous and order == 'C') or
                (array_1.imag.flags.f_contiguous and order == 'F'))
    assert (isinstance(array_1, Array))
    assert (np.may_share_memory(
        array_1, _array) == (not np.iscomplexobj(_array) and
                             get_alignment(_array) >= _DEFAULTS['alignment']))

    # Get new array
    _order = 'F' if order == 'C' else 'C'
    array_1 = asarray(_array, order=_order)

    # Checks
    assert_allclose(array_1, _array)
    assert (isinstance(array_1, Array))
    assert (array_1.dtype == _array.dtype)
    assert ((array_1.real.flags.c_contiguous and _order == 'C') or
            (array_1.real.flags.f_contiguous and _order == 'F'))
    if array_1.iscomplexobj():
        assert ((array_1.imag.flags.c_contiguous and _order == 'C') or
                (array_1.imag.flags.f_contiguous and _order == 'F'))
    assert (isinstance(array_1, Array))
    assert (not np.may_share_memory(array_1, _array))

    # Get new array_1
    _order = 'F' if order == 'C' else 'C'
    _dtype = 'float' if 'complex' in str(dtype) else 'complex'
    array_1 = asarray(_array, dtype=_dtype, order=_order)

    # Checks
    if _dtype == 'float':
        assert_allclose(array_1, _array.real)
    else:
        assert_allclose(array_1, _array.real)
    assert (array_1.dtype == _dtype)
    assert ((array_1.real.flags.c_contiguous and _order == 'C') or
            (array_1.real.flags.f_contiguous and _order == 'F'))
    if array_1.iscomplexobj():
        assert ((array_1.imag.flags.c_contiguous and _order == 'C') or
                (array_1.imag.flags.f_contiguous and _order == 'F'))
    assert (isinstance(array_1, Array))
    assert (not np.may_share_memory(array_1, _array))

    array_1 = array(array_1, dtype=dtype, order=order, alignment=alignment)
    array_2 = array(array_1, copy=False)
    array_3 = array(array_1)  # default, copy == True
    assert (isinstance(array_1, Array))
    assert (isinstance(array_2, Array))
    assert (isinstance(array_3, Array))
    assert (array_1.dtype == array_2.dtype == array_3.dtype == dtype)
    assert (array_1.min_alignment == array_2.min_alignment ==
            array_3.min_alignment == alignment)
    assert (array_1.alignment == array_2.alignment)
    assert (array_1.alignment >= alignment)
    assert (array_3.alignment >= alignment)
    assert (np.may_share_memory(array_1, array_2))
    assert (not np.may_share_memory(array_1, array_3))
    assert (not np.may_share_memory(array_2, array_3))

    array_2 = asarray(array_1)
    assert (isinstance(array_2, Array))
    assert (array_2.dtype == dtype)
    assert (array_2.min_alignment == alignment)
    assert (array_1.alignment == array_2.alignment)
    assert (array_2.alignment >= alignment)
    assert (np.may_share_memory(array_1, array_2))

    array_2 = asarray(array_1, alignment=array_1.alignment)
    assert (isinstance(array_2, Array))
    assert (array_2.dtype == dtype)
    assert (array_2.min_alignment == array_1.alignment)
    assert (array_1.alignment == array_2.alignment)
    assert (np.may_share_memory(array_1, array_2))

    _alignment = max(array_1.real.alignment, array_1.imag.alignment)
    array_2 = asarray(array_1, alignment=2 * _alignment)
    assert (isinstance(array_2, Array))
    assert (array_2.dtype == dtype)
    assert (array_2.min_alignment == 2 * _alignment)
    assert (array_2.alignment >= 2 * _alignment)
    assert (not np.may_share_memory(array_1, array_2))

    # Create array with max alignment == 32
    _ok = False
    for _ in range(100):
        _r = empty(10, alignment=32)
        if _r.alignment == 32:
            _ok = True
            break

    # If couldn't get right alignment, just skip
    if _ok:
        # Create array with min alignment == 64
        _i = empty(10, alignment=64)

        # Get array
        array_1 = Array(buffer=(_r, _i, 32))
        array_2 = asarray(array_1, alignment=32)
        array_3 = asarray(array_1, alignment=64)

        # Checks
        assert (_r.alignment == 32)
        assert (_i.alignment >= 64)
        assert (array_1.min_alignment == 32)
        assert (array_2.min_alignment == 32)
        assert (array_3.min_alignment == 64)
        assert_allclose(array_1, array_2)
        assert_allclose(array_1, array_3)
        assert (np.may_share_memory(array_1, array_2))
        assert (not np.may_share_memory(array_1, array_3))
    else:
        pytest.skip("Too many attempts.")


@pytest.mark.skipif(not load_library('hybridq.so') or
                    not load_library('hybridq_swap.so'),
                    reason="Cannot load HybridQ C++ core")
@pytest.mark.parametrize(
    'dtype,order,alignment',
    [(dtype, order, alignment)
     for dtype in ['complex64', 'complex128', 'complex256'] for order in 'C'
     for alignment in [64, 128, 256, 512, 1024] for _ in range(10)])
def test__tensordot(dtype, order, alignment):
    from hybridq_array.linalg import tensordot
    import warnings

    # Get real type
    rtype = np.dtype(dtype).type(1).real.dtype

    # Get shapes
    shape_a = (2,) * 10
    shape_b = (2,) * 10

    # Get random axes
    axes = np.random.choice(len(shape_b), size=len(shape_a) // 2, replace=False)

    # Get random arrays
    a = np.take(np.random.random(shape_a + (2,)).astype(rtype).view(dtype),
                0,
                axis=-1)
    b = asarray(np.take(np.random.random(shape_b +
                                         (2,)).astype(rtype).view(dtype),
                        0,
                        axis=-1),
                dtype=dtype,
                order=order,
                alignment=alignment)

    # Checks
    assert (isinstance(b, Array))
    assert (b.dtype == dtype)
    assert (b.min_alignment == alignment)
    assert ((b.real.flags.c_contiguous and order == 'C') or
            (b.real.flags.f_contiguous and order == 'F'))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # No warnings/errors should be raised
        array_1 = tensordot(a, b, axes=axes, raise_if_hcore_fails=True)
        assert (not w)

        # No warnings/errors should be raised
        _array_2 = b.copy()
        array_2 = tensordot(a,
                            _array_2,
                            axes=axes,
                            inplace=True,
                            raise_if_hcore_fails=True)
        assert (np.may_share_memory(_array_2, array_2))
        assert (np.may_share_memory(_array_2.real, array_2.real))
        assert (np.may_share_memory(_array_2.imag, array_2.imag))
        assert (not w)

        # A warning should be triggered since the backend is forced
        array_3 = tensordot(a, b, axes=axes, force_backend=True)
        assert (len(w) == 1 and issubclass(w[0].category, UserWarning))
        assert ('HybridQ C++ core is ignored' in str(w[0]))

        # Check
        assert_allclose(array_1, array_3)
        assert_allclose(array_2, array_3)
