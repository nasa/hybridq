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
from hybridq.gate import Gate
from hybridq.circuit.circuit import Circuit
from typing import List, Tuple, Callable, Dict
from hybridq.utils import sort, argsort

# Define Qubit type
Qubit = any

# Define Coupling
Coupling = Tuple[Qubit, Qubit]

# Define Layout
Layout = List[Qubit]


def generate_U(layout: dict[any, list[Coupling]],
               qubits_order: list[Qubit],
               depth: int,
               sequence: list[any],
               one_qb_gates: iter[Gate],
               two_qb_gates: iter[Gate],
               exclude_qubits: iter[Qubit] = None) -> Circuit:
    """
    Generate U at a given depth.
    """

    # Initialize circuit
    circ = Circuit()

    # Get qubits to exclude
    exclude_qubits = set() if exclude_qubits is None else set(exclude_qubits)

    # Remove qubits to exclude from qubits_order
    qubits_order = [q for q in qubits_order if q not in exclude_qubits]

    # Gate index for single-qubit gate
    index = 0

    for d in range(depth):

        # Get sequence
        seq = sequence[d % len(sequence)]

        # Get right layout
        layer = layout[seq]

        # Get tag
        tags = {'depth': d, 'sequence': seq}

        # Add single qubit gates
        circ += [
            next(one_qb_gates).on([q]).set_tags({
                **tags, 'index': index + i
            }) for i, q in enumerate(qubits_order)
        ]

        # Add two qubit gates
        circ += [
            next(two_qb_gates).on(q).set_tags(tags)
            for q in layer
            if not exclude_qubits.intersection(q)
        ]

        # Increment index
        index += len(qubits_order)

    return circ


def generate_OTOC(layout: dict[any, list[Coupling]],
                  depth: int,
                  sequence: list[any],
                  one_qb_gates: iter[Gate],
                  two_qb_gates: iter[Gate],
                  butterfly_op: str,
                  ancilla: Qubit,
                  targets: list[Qubit],
                  qubits_order: list[Qubit] = None) -> Circuit:

    # Get all qubits
    all_qubits = {
        q for s in sequence[:min(depth, len(sequence))] for gate in layout[s]
        for q in gate
    }

    # Get order of qubits
    qubits_order = sort(all_qubits) if qubits_order is None else qubits_order

    # Get list if single butterfly is provided
    butterfly_op = list(butterfly_op)

    # Check order of qubits
    if sort(all_qubits) != sort(qubits_order):
        raise ValueError(
            "'qubits_order' must be a valid permutation of all qubits.")

    # Check if butterfly op has valid strings
    if set(butterfly_op).difference(['I', 'X', 'Y', 'Z']):
        raise ValueError('Only {I, X, Y, Z} are valid butterfly operators')

    # Check if ancilla/targets are in layout
    if set(targets).union([ancilla]).difference(all_qubits):
        raise ValueError(f"Ancilla/Targets must be in layout.")

    # Check if targets are unique
    if len(set(targets)) != len(targets):
        raise ValueError('Targets must be unique.')

    # Check that ancilla is not in targets
    if ancilla in targets:
        raise ValueError('Ancilla must be different from targets')

    # Check if the number of targets corresponds to the number of butterfly ops
    if len(targets) != len(butterfly_op) + 1:
        raise ValueError(
            f"Number of butterfly operators does not match number "
            f"of targets (expected {len(targets)-1}, got {len(butterfly_op)}).")

    # Check that there is a coupling between the ancilla qubit and the measurement qubit
    if next((False for s in sequence[:min(depth, len(sequence))]
             for w in layout[s] if sort(w) == sort([ancilla, targets[0]])),
            True):
        raise ValueError(
            f"No available two-qubit gate between ancilla {ancilla} "
            f"and qubit {targets[0]}.")

    # Initialize Circuit
    circ = Circuit()

    # Add initial layer of single qubit gates
    circ.extend([
        Gate('SQRT_Y' if q != ancilla else 'SQRT_X',
             qubits=[q],
             tags={
                 'depth': 0,
                 'sequence': 'initial'
             }) for q in sort(all_qubits)
    ])

    # Add CZ between ancilla and first target qubit
    circ.append(
        Gate('CZ', [ancilla, targets[0]],
             tags={
                 'depth': 0,
                 'sequence': 'first_control'
             }))

    # Generate U
    U = generate_U(layout=layout,
                   qubits_order=qubits_order,
                   depth=depth,
                   sequence=sequence,
                   one_qb_gates=one_qb_gates,
                   two_qb_gates=two_qb_gates,
                   exclude_qubits=[ancilla]).update_all_tags({'U': True})

    # Add U to circuit
    circ += U

    # Add butterfly operator
    circ.extend([
        Gate(_b,
             qubits=[_t],
             tags={
                 'depth': depth - 1,
                 'sequence': 'butterfly'
             }) for _b, _t in zip(butterfly_op, targets[1:])
    ])

    # Add U* to circuit and update depth
    circ += Circuit(
        gate.update_tags({
            'depth': 2 * depth - gate.tags['depth'] - 1,
            'U^-1': True
        }) for gate in U.inv().remove_all_tags(['U']))

    # Add CZ between ancilla and first target qubit
    circ.append(
        Gate('CZ', [ancilla, targets[0]],
             tags={
                 'depth': 2 * depth - 1,
                 'sequence': 'second_control'
             }))

    return circ
