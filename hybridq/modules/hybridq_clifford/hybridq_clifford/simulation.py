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
import more_itertools as mit
import functools as fts
import itertools as its
import numpy as np
import sys
import os

__all__ = []

# Initialize Pauli matrices
Paulis_ = dict(I=np.eye(2),
               X=np.array([[0, 1], [1, 0]]),
               Y=np.array([[0, -1j], [1j, 0]]),
               Z=np.array([[1, 0], [0, -1]]))
Paulis_.update((i_, Paulis_[x_]) for i_, x_ in enumerate('IXYZ'))


def GetComplexType(dtype: np.dtype) -> np.dtype:
    """
    Convert `dtype` to a complex `dtype`.
    """
    dtype = np.dtype(dtype)
    if np.iscomplexobj(dtype.type(1)):
        return dtype
    else:
        return (np.ones(1, dtype=dtype) + 1j * np.ones(1, dtype=dtype)).dtype


def ToPauliString(x: int, n: int) -> str:
    """
    Convert an integer to a Pauli string.
    """
    return ''.join(map(lambda i: 'IXYZ'[(x // 4**i) % 4], range(n)))


@fts.lru_cache
def GetPauliOperator(op: iter[int] | str, dtype='complex64') -> np.array:
    """
    Given a Pauli string, return its matrix representation.
    """
    return fts.reduce(np.kron, map(lambda g: Paulis_[g].astype(dtype), op))


@fts.lru_cache
def GenerateLinearSystem(n: int, dtype: np.dtype = 'complex64'):
    """
    Generate Pauli's linear system of `n` qubits.
    """
    GPO_ = fts.partial(GetPauliOperator, dtype=dtype)
    TPS_ = fts.partial(ToPauliString, n=n)
    return np.reshape(
        list(map(lambda x: GPO_(TPS_(x)).conj() / 2**n, range(4**n))),
        (4**n, 4**n))


# Get operator decomposition
def DecomposeOperator(gate: np.array,
                      atol: float = 1e-8,
                      dtype: np.dtype = 'float32'):

    # Gate must be a quadratic matrix of size 2^n
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1] or int(
            np.log2(gate.shape[0])) != np.log2(gate.shape[0]):
        raise ValueError("'gate' is not a valid")

    # Get number of qubits
    n_ = int(np.log2(gate.shape[0]))

    # Generate linear system
    LS_ = GenerateLinearSystem(n=n_, dtype=GetComplexType(dtype))

    # Specialize functions
    TPS_ = fts.partial(ToPauliString, n=n_)
    GPO_ = fts.partial(GetPauliOperator, dtype=GetComplexType(dtype))

    # Get decomposition
    gate_ = np.real_if_close(LS_ @ np.array(
        list(
            map(lambda op: (gate @ op @ gate.conj().T).ravel(),
                map(GPO_, map(TPS_, range(4**n_)))))).T).T

    # Find values which are above the given threshold
    pos_ = list(map(lambda x: np.where(np.abs(x) > atol)[0], gate_))

    # Extract
    dec_ = list(map(lambda g, p: g[p], gate_, pos_))

    # Sort so that first state is always the largest
    dec_, pos_ = zip(
        *((d_[s_], p_[s_]) for d_, p_, s_ in ((d_, p_,
                                               np.argsort(np.abs(d_))[::-1])
                                              for d_, p_ in zip(dec_, pos_))))

    # Return
    return dec_, pos_
