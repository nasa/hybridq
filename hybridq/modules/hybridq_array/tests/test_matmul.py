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

from os import environ
from sys import stderr
import numpy as np
import pytest

from hybridq_array.aligned_array import asarray
from hybridq_array import matmul


@pytest.fixture(autouse=True)
def set_seed():
    """
    Set seed for `pytest`.
    """

    # Get random seed
    seed = np.random.randint(2**32 - 1)

    # Get state
    state = np.random.get_state()

    # Set seed
    np.random.seed(seed)

    # Print seed
    print(f"# Used seed [{environ['PYTEST_CURRENT_TEST']}]: {seed}",
          file=stderr)

    # Wait for PyTest
    yield

    # Set state
    np.random.set_state(state)


@pytest.mark.parametrize('dtype,inplace,force_backend',
                         [(dtype, inplace, fb)
                          for dtype in ['complex64', 'complex128']
                          for inplace in [False, True] for fb in [False, True]])
def test_matmul_1(dtype, inplace, force_backend):
    # Fix number of qubits
    n = 10

    # Fix number of axes
    p = 4

    # Get random matrix
    U = np.random.standard_normal((2**p, 2**p)).astype(dtype)
    U += 1j * np.random.standard_normal((2**p, 2**p)).astype(dtype)

    # Get random array
    psi = np.random.standard_normal((2,) * n).astype(dtype)
    psi += 1j * np.random.standard_normal((2,) * n).astype(dtype)

    # Fix axes
    axes = [5, 2, 3, 6]

    # Compute matmul
    psi1 = np.einsum('XYZWxyzw,abyzexwhij->abYZeXWhij',
                     np.reshape(U, (2,) * 2 * p), psi)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=force_backend,
                   raise_if_hcore_fails=True)
    if not inplace or force_backend:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-3)

    # Fix axes
    axes = [5, 2, 3, 7]

    # Compute matmul
    psi1 = np.einsum('XYZWxyzw,abyzexgwij->abYZeXgWij',
                     np.reshape(U, (2,) * 2 * p), psi)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=force_backend,
                   raise_if_hcore_fails=True)
    if not inplace or force_backend:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-3)

    # Fix axes
    axes = [8, 2, 3, 6]

    # Compute matmul
    psi1 = np.einsum('XYZWxyzw,abyzefwhxj->abYZefWhXj',
                     np.reshape(U, (2,) * 2 * p), psi)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=force_backend,
                   raise_if_hcore_fails=True)
    if not inplace or force_backend:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-3)

    # Fix axes
    axes = [9, 2, 3, 6]

    # Compute matmul
    psi1 = np.einsum('XYZWxyzw,abyzefwhix->abYZefWhiX',
                     np.reshape(U, (2,) * 2 * p), psi)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=force_backend,
                   raise_if_hcore_fails=True)
    if not inplace or force_backend:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-3)


@pytest.mark.parametrize('n,p,dtype,inplace,alignment',
                         [(n, p, dtype, inplace, alignment)
                          for n in range(3, 13) for p in range(1, 7)
                          for dtype in ['float32', 'float64']
                          for inplace in [False, True]
                          for alignment in [16, 32]])
def test_matmul_2_cc(n, p, dtype, inplace, alignment):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal(
        (2**p, 2**p)).astype(dtype) + 1j * np.random.standard_normal(
            (2**p, 2**p)).astype(dtype)

    # Get random array
    psi = asarray(np.random.standard_normal(
        (2,) * n).astype(dtype) + 1j * np.random.standard_normal(
            (2,) * n).astype(dtype),
                  alignment=alignment)

    # Get random axes
    axes = np.random.choice(n, size=p, replace=False)

    # Multiply
    psi1 = matmul(U, psi, axes=axes, force_backend=True)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=False,
                   raise_if_hcore_fails=True)
    if not inplace:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-4)


@pytest.mark.parametrize('n,p,dtype,inplace,alignment',
                         [(n, p, dtype, inplace, alignment)
                          for n in range(3, 13) for p in range(1, 7)
                          for dtype in ['float32', 'float64']
                          for inplace in [False, True]
                          for alignment in [16, 32]])
def test_matmul_2_cq(n, p, dtype, inplace, alignment):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal(
        (2**p, 2**p)).astype(dtype) + 1j * np.random.standard_normal(
            (2**p, 2**p)).astype(dtype)

    # Get random array
    psi_re = asarray(np.random.standard_normal((2,) * n).astype(dtype),
                     alignment=alignment)
    psi_im = asarray(np.random.standard_normal((2,) * n).astype(dtype),
                     alignment=alignment)

    # Get random axes
    axes = np.random.choice(n, size=p, replace=False)

    # Multiply
    psi1 = matmul(U, (psi_re, psi_im), axes=axes, force_backend=True)
    psi2 = (psi_re, psi_im)
    psi2_ = matmul(U, (psi_re, psi_im),
                   axes=axes,
                   inplace=inplace,
                   force_backend=False,
                   raise_if_hcore_fails=True)
    if not inplace:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1.real, psi2[0], atol=1e-4)
    np.testing.assert_allclose(psi1.imag, psi2[1], atol=1e-4)


@pytest.mark.parametrize('n,p,dtype', [(n, p, dtype) for n in range(3, 13)
                                       for p in range(1, 7)
                                       for dtype in ['float32', 'float64']
                                       for _ in range(4)])
def test_matmul_2_cr(n, p, dtype):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal(
        (2**p, 2**p)).astype(dtype) + 1j * np.random.standard_normal(
            (2**p, 2**p)).astype(dtype)

    # Get random array
    psi = np.random.standard_normal((2,) * n).astype(dtype)

    # Get random axes
    axes = np.random.choice(n, size=p, replace=False)

    # Multiply
    psi1 = matmul(U, psi, axes=axes, force_backend=True)
    psi2 = matmul(U,
                  psi,
                  axes=axes,
                  force_backend=False,
                  raise_if_hcore_fails=True)

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-4)


@pytest.mark.parametrize('n,p,dtype,inplace,alignment',
                         [(n, p, dtype, inplace, alignment)
                          for n in range(3, 13) for p in range(1, 7)
                          for dtype in ['float32', 'float64']
                          for inplace in [False, True]
                          for alignment in [16, 32]])
def test_matmul_2_rc(n, p, dtype, inplace, alignment):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal((2**p, 2**p)).astype(dtype)

    # Get random array
    psi = asarray(np.random.standard_normal(
        (2,) * n).astype(dtype) + 1j * np.random.standard_normal(
            (2,) * n).astype(dtype),
                  alignment=alignment)

    # Get random axes
    axes = np.random.choice(n, size=p, replace=False)

    # Multiply
    psi1 = matmul(U, psi, axes=axes, force_backend=True)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=False,
                   raise_if_hcore_fails=True)
    if not inplace:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-4)


@pytest.mark.parametrize('n,p,dtype,inplace,alignment',
                         [(n, p, dtype, inplace, alignment)
                          for n in range(3, 13) for p in range(1, 7)
                          for dtype in ['float32', 'float64']
                          for inplace in [False, True]
                          for alignment in [16, 32]])
def test_matmul_2_rr(n, p, dtype, inplace, alignment):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal((2**p, 2**p)).astype(dtype)

    # Get random array
    psi = asarray(np.random.standard_normal((2,) * n).astype(dtype),
                  alignment=alignment)

    # Get random axes
    axes = np.random.choice(n, size=p, replace=False)

    # Multiply
    psi1 = matmul(U, psi, axes=axes, force_backend=True)
    psi2 = psi.copy()
    psi2_ = matmul(U,
                   psi2,
                   axes=axes,
                   inplace=inplace,
                   force_backend=False,
                   raise_if_hcore_fails=True)
    if not inplace:
        psi2 = psi2_

    # Check
    np.testing.assert_allclose(psi1, psi2, atol=1e-4)
