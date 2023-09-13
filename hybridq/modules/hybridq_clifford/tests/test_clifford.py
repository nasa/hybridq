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

import itertools as its
import functools as fts
import numpy as np
import pytest


# Exand gate to the right number of qubits
def expand_gate_(gate, n_qubits):
    # Split matrix and qubits
    gate_, qubits_ = gate

    # Check number of qubits
    assert (len(qubits_) <= n_qubits)

    # Add missing qubits to matrix
    gate_ = np.reshape(np.kron(gate_, np.eye(2**(n_qubits - len(qubits_)))),
                       (2,) * 2 * n_qubits)

    # Add missing qubits
    qubits_ = tuple(qubits_) + tuple(
        q_ for q_ in range(n_qubits) if q_ not in qubits_)
    qubits_ = qubits_ + tuple(q_ + n_qubits for q_ in qubits_)

    # Reorder qubits and return expanded matrix
    return np.reshape(
        np.transpose(gate_, [qubits_.index(q_) for q_ in range(2 * n_qubits)]),
        (2**n_qubits, 2**n_qubits))


# Get density matrix from stabilizers
def density_matrix_(stabs, phases):
    from hybridq_clifford.simulation import GetPauliOperator_
    from functools import reduce

    # Get number of qubits
    n_qubits_ = len(stabs[0])

    # Get all pauli operators
    ops_ = list(
        map(
            lambda x, y: (np.eye(2**n_qubits_) + x * y), phases,
            map(GetPauliOperator_,
                map(lambda x: ''.join('IXYZ'[x] for x in x), stabs))))

    # Return density matrix
    return reduce(np.matmul, ops_) / 2**n_qubits_


# Pi/2 rotations on the XY plane
r_pi_2_ = np.array([[[(0.707106781187 + 0j), (-0 - 0.707106781187j)],
                     [-0.707106781187j, (0.707106781187 + 0j)]],
                    [[(0.707106781187 + 0j), (-0.707106781187 - 0j)],
                     [(0.707106781187 - 0j), (0.707106781187 + 0j)]],
                    [[(0.707106781187 + 0j), (-0 + 0.707106781187j)],
                     [0.707106781187j, (0.707106781187 + 0j)]],
                    [[(0.707106781187 + 0j), (0.707106781187 + 0j)],
                     [(-0.707106781187 + 0j), (0.707106781187 + 0j)]]])

# Control Z
cz_ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]],
               dtype=complex)


def get_clifford_rqc_(n_qubits, n_cycles, seed=None):
    # Initialize random number generator
    rng_ = np.random.default_rng(seed)

    # Single layer
    get_layer_ = lambda: its.chain(
        (map(lambda x: (r_pi_2_[x[1]], [x[0]]),
             zip(range(n_qubits), rng_.integers(4, size=n_qubits)))), (map(
                 lambda qs: (cz_, qs),
                 filter(lambda x: x[0] < x[1],
                        its.product(range(n_qubits), range(n_qubits))))))

    # Return circuit
    return list(fts.reduce(its.chain, [get_layer_() for _ in range(n_cycles)]))


# Simple class to mimic a stabilizer
class Stab_:

    def __init__(self, paulis, phase=1):
        self.paulis = paulis
        self.phase = phase

    def __str__(self):
        return f"({self.phase}, {''.join('IXYZ'[x_] for x_ in self.paulis)})"

    def __repr__(self):
        return str(self)

    def __mul__(self, other):
        from hybridq_clifford.utils import mat_p_, mat_s_
        paulis_, phases_ = zip(*map(lambda x, y: (mat_p_[x, y], mat_s_[x, y]),
                                    self.paulis, other.paulis))
        return Stab_(paulis_, self.phase * other.phase * np.prod(phases_))

    def __eq__(self, other):
        return np.allclose(self.paulis, other.paulis) and np.isclose(
            self.phase, other.phase)


def test_PauliString(seed=None):
    from hybridq_clifford.simulation import (ToPauliString, FromPauliString,
                                             ToPauliStringFromState_,
                                             StateFromPauliString_,
                                             GetSubState_, UpdateState_)
    from numba.typed import List as NumbaList

    # If seed is None, initialize it with a new seed
    seed = np.random.randint(2**32) if seed is None else seed

    # Get new rng
    rng_ = np.random.default_rng(seed)

    assert (all(
        map(lambda p: ToPauliString(FromPauliString(p), len(p)) == p,
            map(''.join, rng_.choice(list('IXYZ'), size=(100, 20))))))
    assert (all(
        map(lambda x: FromPauliString(ToPauliString(x, 20)) == x,
            rng_.integers(4**20, size=100))))

    assert (all(
        map(
            lambda x: ToPauliStringFromState_(StateFromPauliString_(x),
                                              n_paulis=len(x)) == x,
            map(''.join, rng_.choice(list('IXYZ'), size=(1000, 50))))))
    assert (np.all(
        list(
            map(
                lambda x: StateFromPauliString_(
                    ToPauliStringFromState_(x, 4 * len(x))) == x,
                rng_.integers(2**8, size=(1000, 10))))))

    assert (all(
        ToPauliString(GetSubState_(StateFromPauliString_(x), q), len(q)) ==
        ''.join(x[q] for q in q) for x, q in zip(
            map(''.join, rng_.choice(list('IXYZ'), size=(1000, 50))), (
                rng_.choice(50, size=11, replace=False) for _ in range(1000)))))
    assert (all(
        ToPauliStringFromState_(
            UpdateState_(StateFromPauliString_(x), FromPauliString(y), q), len(
                x)) == ''.join(y[q.index(i_)] if i_ in q else x[i_]
                               for i_ in range(len(x)))
        for x, y, q in
        zip(map(''.join, rng_.choice(list('IXYZ'), size=(
            1000, 50))), map(''.join, rng_.choice(list('IXYZ'), size=(
                1000, 11))), (map(lambda x: NumbaList(x), (
                    rng_.choice(50, size=11, replace=False)
                    for _ in range(1000)))))))


@pytest.mark.parametrize('n_qubits,n_gates,parallel',
                         [(6, 4, False)] * 5 + [(6, 4, True)] * 5)
def test_Simulation(n_qubits, n_gates, parallel, verbose=False, seed=None):
    from hybridq_clifford.simulation import simulate, GetPauliOperator_, ToPauliStringFromState_
    from os import cpu_count

    # If seed is None, initialize it with a new seed
    seed = np.random.randint(2**32) if seed is None else seed

    # Get new rng
    rng_ = np.random.default_rng(seed)

    # Generate random gate with a given number of qubits
    def random_gate_(n):
        gate_ = rng_.random((2**n, 2**n)) + 1j * rng_.random((2**n, 2**n))
        return gate_ / np.linalg.norm(gate_.ravel())

    # Generate random circuit
    circuit_ = [(random_gate_(n_), rng_.choice(n_qubits, size=n_,
                                               replace=False))
                for n_ in rng_.integers(1, 4, size=n_gates)]

    # Get a random initial pauli string
    paulis_ = ''.join(rng_.choice(list('XYZ'), size=n_qubits))

    # Get all branches
    all_, all_info_ = simulate(circuit_,
                               paulis_,
                               verbose=verbose,
                               parallel=parallel)

    # Reconstruct density matrix for clifford
    RhoClifford_ = sum(
        map(
            lambda x: x[1] * GetPauliOperator_(
                ToPauliStringFromState_(x[0], n_qubits)), all_.items()))

    # Reconstruct density matrix from exact
    RhoExact_ = fts.reduce(
        lambda x, y: y @ x,
        map(fts.partial(expand_gate_, n_qubits=n_qubits), circuit_))
    RhoExact_ = RhoExact_ @ GetPauliOperator_(paulis_) @ RhoExact_.conj().T

    # Check
    np.testing.assert_allclose(RhoClifford_, RhoExact_, atol=1e-4)


@pytest.mark.parametrize('n_qubits', [8] * 5)
def test_Clifford(n_qubits):
    from hybridq_clifford.simulation import simulate, ToPauliStringFromState_, Paulis_
    from hybridq_clifford.utils import diag_z_, trace_, mat_p_, mat_s_, mul_

    # Check that gates are really clifford
    assert (len(
        simulate(list(zip(r_pi_2_, its.repeat([0]))) + [(cz_, [0, 1])],
                 'XY')[0]) == 1)

    # Check multiplication matrices for paulis
    assert (all(
        np.allclose(Paulis_['IXYZ'[i_]] @ Paulis_['IXYZ'[j_]], mat_s_[i_, j_] *
                    Paulis_['IXYZ'[mat_p_[i_, j_]]])
        for i_ in range(4)
        for j_ in range(4)))

    # Check stabilizer multiplication
    for _ in range(1000):
        s1_ = Stab_(np.random.randint(4, size=200), np.random.random())
        s2_ = Stab_(np.random.randint(4, size=200), np.random.random())
        assert (Stab_(
            *mul_(s1_.paulis, s2_.paulis.copy(), s1_.phase, s2_.phase)) == s1_ *
                s2_)

    # Initial state is all '+'
    paulis_ = {
        ''.join('X' if i_ == j_ else 'I'
                for j_ in range(n_qubits)): 1
        for i_ in range(n_qubits)
    }

    # Generate random clifford
    circuit_ = get_clifford_rqc_(n_qubits=n_qubits, n_cycles=30)

    # Simulate circuit using clifford expansion
    stabs_ = simulate(circuit_, paulis_, parallel=False)[0]

    # Convert stabilizers
    stabs_ = dict(
        map(lambda x: (ToPauliStringFromState_(x[0], n_qubits), x[1]),
            stabs_.items()))

    # Convert phases and stabilizers
    phases_, rho_ = map(
        np.asarray,
        zip(*map(
            lambda p:
            (complex(p[1]),
             np.fromiter(map(lambda x: dict(I=0, X=1, Y=2, Z=3)[x], p[0]),
                         dtype='uint8')), stabs_.items())))

    # Simulate exactly
    psi_ = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    for U_ in map(lambda g: expand_gate_(g, n_qubits), circuit_):
        psi_ = U_ @ psi_

    # Get exact density matrix
    ex_rho_ = np.reshape(np.kron(psi_.ravel(),
                                 psi_.ravel().conj()),
                         (2**n_qubits, 2**n_qubits))

    # Check density matrices
    np.testing.assert_allclose(ex_rho_,
                               density_matrix_(rho_, phases_),
                               atol=1e-5)

    # Get stabilizers which are in the Z-base
    z_rho_, z_phases_ = diag_z_(rho_.copy(), phases_.copy())

    if len(z_phases_):
        # Stabilizers in the Z-base should be diagonal
        assert (np.all(
            np.subtract(*np.where(
                np.abs(density_matrix_(z_rho_, z_phases_)) > 1e-6)) == 0))

        # Stabilizers in the Z-base should correspond to |psi|^2
        np.testing.assert_allclose(np.diag(density_matrix_(z_rho_, z_phases_)),
                                   np.abs(psi_.ravel())**2,
                                   atol=1e-6)
    else:
        np.testing.assert_allclose(np.abs(psi_.ravel())**2,
                                   1 / 2**n_qubits,
                                   atol=1e-4)

    # Check inverse participation ratio
    assert (np.isclose(np.log2(psi_.size * np.sum(np.abs(psi_)**4)),
                       len(z_rho_),
                       atol=1e-5))
