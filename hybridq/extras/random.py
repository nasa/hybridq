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
from hybridq.gate.utils import get_available_gates, get_clifford_gates
from hybridq.gate import Gate, MatrixGate
from hybridq.circuit import Circuit
import numpy as np


def get_indexes(n_qubits: int, *, use_random_indexes: bool = False):
    # Initialize
    indexes = []

    # Randomize indexes
    if use_random_indexes:

        # Add strings
        indexes = []
        while len(indexes) < n_qubits // 3:
            indexes += [
                ''.join(
                    np.random.choice(list('abcdefghijklmnopqrstuvwxyz'),
                                     size=20))
                for _ in range(n_qubits // 3 - len(indexes))
            ]

        # Add tuples
        while len(indexes) < n_qubits:
            indexes += [
                tuple(x) for x in np.unique(np.random.randint(
                    -2**32 + 1, 2**32 - 1, size=(n_qubits - len(indexes), 2)),
                                            axis=0)
            ]

        # Random permutation
        indexes = [indexes[i] for i in np.random.permutation(n_qubits)]

    # Use sequential
    else:
        indexes = np.arange(n_qubits)

    # Return indexes
    return indexes


def get_random_gate(randomize_power: bool = True,
                    use_clifford_only: bool = False,
                    use_unitary_only: bool = True):
    """
    Generate random gate.
    """
    # Get available gates
    avail_gates = get_clifford_gates(
    ) if use_clifford_only else get_available_gates()

    # Add random matrices
    if not use_unitary_only:
        avail_gates = avail_gates + ('RANDOM_MATRIX',)

    # Get random gate
    gate_name = np.random.choice(avail_gates)

    # Generate a random matrix
    if gate_name == 'RANDOM_MATRIX':
        # Get random number of qubits
        n_qubits = np.random.choice(range(1, 3))

        # Get random matrix
        M = 2 * np.random.random(
            (2**n_qubits, 2**n_qubits)).astype('complex') - 1
        M += 1j * (2 * np.random.random((2**n_qubits, 2**n_qubits)) - 1)
        M /= 2

        # Get gate
        gate = MatrixGate(M)

    # Generate named gate
    else:
        gate = Gate(gate_name)

    # Apply random parameters if present
    if gate.provides('params'):
        gate._set_params(np.random.random(size=gate.n_params))

    # Apply random power
    gate = gate**(2 * np.random.random() - 1 if randomize_power else 1)

    # Apply conjugation if supported
    if gate.provides('conj') and np.random.random() < 0.5:
        gate._conj()

    # Apply transposition if supported
    if gate.provides('T') and np.random.random() < 0.5:
        gate._T()

    # Convert to MatrixGate half of the times
    gate = gate if gate.name == 'MATRIX' or np.random.random(
    ) < 0.5 else MatrixGate(gate.matrix())

    # Return gate
    return gate


def get_rqc(n_qubits: int,
            n_gates: int,
            *,
            indexes: list[int] = None,
            randomize_power: bool = True,
            use_clifford_only: bool = False,
            use_unitary_only: bool = True,
            use_random_indexes: bool = False,
            verbose: bool = False):
    """
  Generate random quantum circuit.
  """
    from tqdm.auto import tqdm

    # Initialize circuit
    circuit = Circuit()

    # If not provided, generate indexes
    indexes = get_indexes(n_qubits, use_random_indexes=use_random_indexes
                         ) if indexes is None else list(indexes)

    # Check that size is correct
    assert (len(indexes) == n_qubits)

    # Get random gates
    gates = (get_random_gate(randomize_power=randomize_power,
                             use_unitary_only=use_unitary_only,
                             use_clifford_only=use_clifford_only)
             for _ in range(n_gates))

    # Assign random qubits, and return circuit
    return Circuit(
        gate.on([
            indexes[i]
            for i in np.random.choice(n_qubits, gate.n_qubits, replace=False)
        ])
        for gate in tqdm(gates,
                         disable=not verbose,
                         total=n_gates,
                         desc='Generating random circuit'))
