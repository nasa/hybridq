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


@pytest.mark.parametrize('n,p,dtype,inplace',
                         [(n, p, dtype, inplace) for n in range(3, 13)
                          for p in range(1, 7)
                          for dtype in ['complex64', 'complex128']
                          for inplace in [False, True] for _ in range(10)])
def test_matmul_2(n, p, dtype, inplace):
    # p cannot be larger than n!
    p = min(p, n)

    # Get random matrix
    U = np.random.standard_normal((2**p, 2**p)).astype(dtype)
    U += 1j * np.random.standard_normal((2**p, 2**p)).astype(dtype)

    # Get random array
    psi = np.random.standard_normal((2,) * n).astype(dtype)
    psi += 1j * np.random.standard_normal((2,) * n).astype(dtype)

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
