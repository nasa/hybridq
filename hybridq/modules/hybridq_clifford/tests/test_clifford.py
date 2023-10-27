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

from scipy.stats import unitary_group
import itertools as its
import functools as fts
import numpy as np
import pytest

# Get random unitaries
random_unitary = unitary_group.rvs


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
    from hybridq_clifford.simulation import GetPauliOperator
    from functools import reduce

    # Get number of qubits
    n_qubits_ = len(stabs[0])

    # Get all pauli operators
    ops_ = list(
        map(
            lambda x, y: (np.eye(2**n_qubits_) + x * y), phases,
            map(GetPauliOperator,
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


def test_Pauli():
    from hybridq_clifford.core import (PauliFromState, StateFromPauli, GetPauli,
                                       SetPauli, SetPauliFromChar)

    for _ in range(100):

        # Check Pauli -> State -> Pauli
        assert (all(
            PauliFromState(StateFromPauli(x_)) == x_
            for x_ in (''.join(x_)
                       for x_ in np.random.choice(list('IXYZ'), size=(100,
                                                                      100)))))

        # Check State -> Pauli -> State
        assert (all(
            StateFromPauli(PauliFromState(x_)) == x_
            for x_ in np.random.randint(2, size=(100, 100))))

        # Check that the right Pauli is returned
        assert (all(
            list(map(lambda i: GetPauli(s_, i), range(len(s_) // 2))) == list(
                map('IXYZ'.index, x_))
            for s_, x_ in ((StateFromPauli(x_), x_) for x_ in (
                ''.join(x_)
                for x_ in np.random.choice(list('IXYZ'), size=(100, 100))))))

        for _ in range(100):
            # Get random string
            x_ = ''.join(np.random.choice(list('IXYZ'), size=100))

            # Check SetPauliFromChar
            s_ = StateFromPauli('I' * 100)
            for i_, c_ in enumerate(x_):
                SetPauliFromChar(s_, i_, c_)
            assert (PauliFromState(s_) == x_)

            # Check SetPauli
            s_ = StateFromPauli('I' * 100)
            for i_, c_ in enumerate(x_):
                SetPauli(s_, i_, 'IXYZ'.index(c_))
            assert (PauliFromState(s_) == x_)


@pytest.mark.parametrize('n_qubits,n_gates,parallel',
                         [(6, 6, False)] * 3 + [(6, 6, True)] * 3)
def test_Simulation(n_qubits, n_gates, parallel, *, verbose=False):
    from hybridq_clifford.simulation import (simulate, GetPauliOperator)
    from hybridq_clifford.core import PauliFromState

    # Get random gate
    def rg_():
        return random_unitary(2**2), np.random.choice(n_qubits,
                                                      size=2,
                                                      replace=False)

    # Get random circuit
    circ_ = [rg_() for _ in range(n_gates)]

    # Get random initial state
    initial_state_ = ''.join(np.random.choice(list('IXYZ'), size=n_qubits))

    # Simulate using clifford expansion
    info_, branches_ = simulate(circuit=circ_,
                                paulis=initial_state_,
                                parallel=parallel,
                                verbose=verbose)

    # Check all branches are correctly split among the different buckets
    ps_ = [set(map(PauliFromState, br_.keys())) for br_ in branches_]
    assert all(
        len(set(ps_[i_]).intersection(ps_[j_])) == 0
        for i_ in range(len(ps_))
        for j_ in range(len(ps_))
        if i_ != j_)

    # Get final density matrix
    rho_exp_ = sum(
        sum(
            map(lambda x: x[1] * GetPauliOperator(PauliFromState(x[0])),
                br_.items())) for br_ in branches_)

    # Simulate using state vector
    rho_sv_ = GetPauliOperator(initial_state_)
    for c_ in circ_:
        U_ = expand_gate_(c_, n_qubits)
        rho_sv_ = U_ @ rho_sv_ @ U_.conj().T

    # Check
    np.testing.assert_allclose(rho_exp_, rho_sv_, atol=1e-4)


@pytest.mark.parametrize('n_qubits', [8] * 5)
def test_Clifford(n_qubits, seed=None):
    from hybridq_clifford.simulation import (simulate, Paulis_)
    from hybridq_clifford.core import PauliFromState
    from hybridq_clifford.utils import diag_z_, trace_, mat_p_, mat_s_, mul_

    # Initialize prng
    rng_ = np.random.default_rng(seed)

    # Check that gates are really clifford
    assert (len(
        simulate(list(zip(r_pi_2_, its.repeat([0]))) + [(cz_, [0, 1])],
                 'XY')[1]))

    # Check multiplication matrices for paulis
    assert (all(
        np.allclose(Paulis_['IXYZ'[i_]] @ Paulis_['IXYZ'[j_]], mat_s_[i_, j_] *
                    Paulis_['IXYZ'[mat_p_[i_, j_]]])
        for i_ in range(4)
        for j_ in range(4)))

    # Check stabilizer multiplication
    for _ in range(1000):
        s1_ = Stab_(rng_.integers(4, size=200), rng_.random())
        s2_ = Stab_(rng_.integers(4, size=200), rng_.random())
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
    circuit_ = get_clifford_rqc_(n_qubits=n_qubits,
                                 n_cycles=30,
                                 seed=rng_.integers(2**32 - 1))

    # Simulate circuit using clifford expansion
    stabs_ = simulate(circuit_,
                      paulis_,
                      parallel=False,
                      norm_atol=1e-1,
                      log2_n_buckets=0)[1]

    # Check number of final stabilizers
    assert len(stabs_) == 1 and len(stabs_[0]) == n_qubits

    # Convert stabilizers
    stabs_ = {
        PauliFromState(x_[0]): int(np.round(x_[1])) for x_ in stabs_[0].items()
    }

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

    # Check diagonal for exact density matrix
    np.testing.assert_allclose(np.diag(ex_rho_), np.abs(psi_)**2, atol=1e-5)

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
