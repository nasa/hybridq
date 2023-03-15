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

from functools import partial as partial_fn
import numpy as np
import pytest

from hybridq_array.aligned_array import get_alignment, array, asarray
from hybridq_array.aligned_array import empty, empty_like
from hybridq_array.aligned_array import zeros, zeros_like
from hybridq_array.aligned_array import ones, ones_like

# Define assert_allclose
assert_allclose = partial_fn(np.testing.assert_allclose, atol=1e-7)


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
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__AlignedArray(dtype, order, alignment):
    from hybridq_array.aligned_array import AlignedArray

    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get new aligned array
    array = empty(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Get random array
    r = (100 * np.random.random(size=shape)).astype(dtype)

    # Assign
    array[:] = r

    # Checks
    assert isinstance(array, AlignedArray)
    np.testing.assert_allclose(array, r)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment >= alignment
    assert array.alignment == get_alignment(array)
    assert array.isaligned(alignment // 2)
    assert array.isaligned(alignment)
    assert array.isaligned(get_alignment(array))
    assert not array.isaligned(2 * get_alignment(array))

    # Check align (default, copy == True)
    if ((alignment // 2) % np.dtype(dtype).itemsize) == 0:
        _array = array.align(alignment // 2)
        assert not np.may_share_memory(array, _array)
        np.testing.assert_allclose(array, _array)
    #
    _array = array.align(alignment)
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.align(array.alignment)
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.align(2 * array.alignment)
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)

    # Check align (copy == False)
    if ((alignment // 2) % np.dtype(dtype).itemsize) == 0:
        _array = array.align(alignment // 2, copy=False)
        assert np.may_share_memory(array, _array)
        np.testing.assert_allclose(array, _array)
    #
    _array = array.align(alignment, copy=False)
    assert np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.align(array.alignment, copy=False)
    assert np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.align(2 * array.alignment, copy=False)
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)

    # Check copy
    _array = array.copy()
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.copy(alignment=2 * alignment)
    assert _array.alignment >= 2 * alignment
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)

    # Check astype (default, copy == True)
    _array = array.astype('complex' if not 'complex' in dtype else 'float',
                          alignment=2 * array.alignment)
    assert _array.dtype == 'complex' if not 'complex' in dtype else 'float'
    assert _array.alignment >= 2 * array.alignment
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.astype(dtype, alignment=2 * array.alignment)
    assert _array.dtype == array.dtype
    assert _array.alignment >= 2 * array.alignment
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    if ((alignment // 2) % np.dtype(dtype).itemsize) == 0:
        _array = array.astype(dtype, alignment=array.alignment // 2)
        assert array.alignment >= array.alignment // 2
        assert not np.may_share_memory(array, _array)
        np.testing.assert_allclose(array, _array)

    # Check astype (default, copy == False)
    _array = array.astype('complex' if not 'complex' in dtype else 'float',
                          alignment=2 * array.alignment,
                          copy=False)
    assert _array.dtype == 'complex' if not 'complex' in dtype else 'float'
    assert _array.alignment >= 2 * array.alignment
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.astype(dtype, alignment=2 * array.alignment, copy=False)
    assert _array.dtype == array.dtype
    assert _array.alignment >= 2 * array.alignment
    assert not np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.astype(dtype, copy=False)
    assert _array.alignment == array.alignment
    assert np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)
    #
    _array = array.astype(dtype, alignment=array.alignment, copy=False)
    assert _array.alignment == array.alignment
    assert np.may_share_memory(array, _array)
    np.testing.assert_allclose(array, _array)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__empty_zeros_ones(dtype, order, alignment):
    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random
    r = (100 * np.random.random(size=shape)).astype(dtype, order=order)

    # Get new aligned array
    array = empty(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Checks
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment >= alignment and array.alignment == get_alignment(
        array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)

    # Get new aligned array
    array = empty_like(r)

    # Checks
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment == get_alignment(array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)

    # Get new zeros array
    array = zeros(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Checks
    np.testing.assert_allclose(array, 0)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment >= alignment and array.alignment == get_alignment(
        array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)

    # Get new ones array
    array = ones(shape=shape, dtype=dtype, order=order, alignment=alignment)

    # Checks
    np.testing.assert_allclose(array, 1)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment >= alignment and array.alignment == get_alignment(
        array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)

    # Get new zeros array
    array = zeros_like(r)

    # Checks
    np.testing.assert_allclose(array, 0)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment == get_alignment(array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)

    # Get new ones array
    array = ones_like(r)

    # Checks
    np.testing.assert_allclose(array, 1)
    assert array.shape == shape
    assert array.dtype == dtype
    assert array.alignment == get_alignment(array)
    assert (order == 'C' and
            array.flags.c_contiguous) or (order == 'F' and
                                          array.flags.f_contiguous)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__array(dtype, order, alignment):
    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random
    r = (100 * np.random.random(size=shape)).astype(dtype, order=order)

    # Get array (default, copy == True)
    a = array(r, dtype=dtype, order=order, alignment=alignment)

    # Checks
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.alignment >= alignment and a.alignment == get_alignment(a)
    assert (order == 'C' and a.flags.c_contiguous) or (order == 'F' and
                                                       a.flags.f_contiguous)
    assert not np.may_share_memory(a, r)
    np.testing.assert_allclose(a, r)

    # Get array (copy == False)
    a = array(r,
              dtype=dtype,
              order=order,
              alignment=2 * get_alignment(r),
              copy=False)

    # Checks
    assert a.shape == shape
    assert a.dtype == dtype
    assert a.alignment >= 2 * get_alignment(r) and a.alignment == get_alignment(
        a)
    assert (order == 'C' and a.flags.c_contiguous) or (order == 'F' and
                                                       a.flags.f_contiguous)
    assert not np.may_share_memory(a, r)
    np.testing.assert_allclose(a, r)

    # Get array (copy == False)
    if (get_alignment(r) % np.dtype(dtype).itemsize) == 0:
        a = array(r,
                  dtype=dtype,
                  order=order,
                  alignment=get_alignment(r),
                  copy=False)

        # Checks
        assert a.shape == shape
        assert a.dtype == dtype
        assert a.alignment == get_alignment(r) and a.alignment == get_alignment(
            a)
        assert (order == 'C' and a.flags.c_contiguous) or (order == 'F' and
                                                           a.flags.f_contiguous)
        assert np.may_share_memory(a, r)
        np.testing.assert_allclose(a, r)


@pytest.mark.parametrize(
    'dtype,order,alignment', [(dtype, order, alignment) for dtype in [
        'float32', 'float64', 'int16', 'int32', 'int64', 'uint16', 'uint32',
        'uint64', 'complex64', 'complex128', 'complex256'
    ] for order in 'CF' for alignment in [32, 64, 128, 256, 512, 1024]])
def test__asarray(dtype, order, alignment):
    # Get random shape
    shape = tuple(np.random.randint(2, 5, size=4))

    # Get random
    r = (100 * np.random.random(size=shape)).astype(dtype, order=order)

    # If numpy array and alignment == None (default, alignment == None)
    array = asarray(r)

    # Check
    assert np.may_share_memory(array, r)

    # If alignment is provided, may create new
    array = asarray(r, alignment=2 * get_alignment(r))

    # Check
    assert not np.may_share_memory(array, r)


@pytest.mark.parametrize('dtype,order,alignment',
                         [(dtype, order, alignment) for dtype in [
                             'float32', 'float64', 'float128', 'int8', 'int16',
                             'int32', 'int64', 'uint8', 'uint16', 'uint32',
                             'uint64', 'complex64', 'complex128', 'complex256'
                         ] for order in 'C'
                          for alignment in [32, 64, 128, 256, 512, 1024]])
def test__transpose(dtype, order, alignment):
    from hybridq_array.aligned_array import AlignedArray
    from hybridq_array import transpose

    # Get random shape
    shape = (2,) * 10

    # Get random
    array = asarray((100 * np.random.random(size=shape)).astype(dtype,
                                                                order=order),
                    alignment=alignment)

    # Checks
    assert array.dtype == dtype
    assert (array.flags.c_contiguous and
            order == 'C') or (array.flags.f_contiguous and order == 'F')
    assert array.alignment >= alignment

    try:
        # Transpose (default, inplace == False)
        _axes = np.arange(array.ndim)
        _axes[-8:] = np.random.permutation(_axes[-8:])
        _array_1 = transpose(array, _axes, raise_if_hcore_fails=True)
        _array_2 = transpose(array, _axes, force_backend=True)
        assert isinstance(_array_1, AlignedArray)
        assert not np.may_share_memory(_array_1, array)
        assert _array_1.dtype == dtype
        assert_allclose(_array_1, _array_2)

        # Transpose (inplace == True)
        _axes = np.arange(array.ndim)
        _axes[-8:] = np.random.permutation(_axes[-8:])
        _array = array.copy()
        _array_1 = transpose(_array,
                             _axes,
                             raise_if_hcore_fails=True,
                             inplace=True)
        _array_2 = transpose(array, _axes, force_backend=True)
        assert isinstance(_array_1, AlignedArray)
        assert not np.may_share_memory(_array_1, array)
        assert np.may_share_memory(_array_1, _array)
        assert not np.may_share_memory(_array_2, _array)
        assert _array_1.dtype == dtype
        assert_allclose(_array_1, _array_2)
        assert_allclose(_array_1, _array)

    except AssertionError as e:
        if not (np.iscomplexobj(array) or order == 'F' or
                str(dtype) in ['int16', 'uint16']):
            raise e

    # Transpose (default, inplace == False)
    _axes = np.arange(array.ndim)
    _axes[-8:] = np.random.permutation(_axes[-8:])
    _array_1 = transpose(array, _axes)
    _array_2 = transpose(array, _axes, force_backend=True)
    assert isinstance(_array_1, AlignedArray)
    assert _array_1.dtype == dtype
    assert_allclose(_array_1, _array_2)

    # Transpose (inplace == True)
    _axes = np.arange(array.ndim)
    _axes[-8:] = np.random.permutation(_axes[-8:])
    _array = array.copy()
    _array_1 = transpose(_array, _axes, inplace=True)
    _array_2 = transpose(array, _axes, force_backend=True)
    assert isinstance(_array_1, AlignedArray)
    assert _array_1.dtype == dtype
    assert_allclose(_array_1, _array_2)
