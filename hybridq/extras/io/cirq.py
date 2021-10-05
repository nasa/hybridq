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
from hybridq.circuit import Circuit
from hybridq.gate import Gate
from hybridq.gate.property import QubitGate, ParamGate
from tqdm.auto import tqdm
import numpy as np
import cirq


def to_cirq(circuit: Circuit,
            qubits_map: dict[any, any] = None,
            verbose: bool = False) -> cirq.Circuit:
    """
    Convert `Circuit` to `cirq.Circuit`.

    Parameters
    ----------
    circuit: Circuit
        `Circuit` to convert to `cirq.Circuit`.
    qubits_map: dict[any, any], optional
        How to map qubits in `Circuit` to `cirq.Circuit`. if not provided,
        `cirq.LineQubit`s are used and automatically mapped to `Circuit`'s
        qubits.
    verbose: bool, optional
        Verbose output.

    Returns
    -------
    cirq.Circuit
        `cirq.Circuit` obtained from `Circuit`.

    Example
    -------
    >>> from hybridq.extras.cirq import to_cirq
    >>> c = Circuit(Gate('H', qubits=[q]) for q in range(3))
    >>> c.append(Gate('CX', qubits=[0, 1]))
    >>> c.append(Gate('CX', qubits=[2, 0]))
    >>> c.append(Gate('CX', qubits=[1, 2]))
    >>> to_cirq(c)
    0: ───H───@───X───────
              │   │
    1: ───H───X───┼───@───
                  │   │
    2: ───H───────@───X───
    """

    _to_cirq_naming = {
        'I': lambda params: cirq.I,
        'P': lambda params: cirq.S,
        'T': lambda params: cirq.T,
        'X': lambda params: cirq.X,
        'Y': lambda params: cirq.Y,
        'Z': lambda params: cirq.Z,
        'H': lambda params: cirq.H,
        'RX': lambda params: cirq.rx(*params),
        'RY': lambda params: cirq.ry(*params),
        'RZ': lambda params: cirq.rz(*params),
        'CZ': lambda params: cirq.CZ,
        'ZZ': lambda params: cirq.ZZ,
        'CX': lambda params: cirq.CNOT,
        'SWAP': lambda params: cirq.SWAP,
        'ISWAP': lambda params: cirq.ISWAP,
        'FSIM': lambda params: cirq.FSimGate(*params),
        'CPHASE': lambda params: cirq.CZ**(params[0] / np.pi)
    }

    # Get circuit
    circ = cirq.Circuit()

    # If not provided, create trivial map
    if not qubits_map:
        try:
            sorted(circuit.all_qubits())
            _standard_sortable = True
        except:
            _standard_sortable = False

        if _standard_sortable:
            qubits_map = {q: cirq.LineQubit(q) for q in circuit.all_qubits()}
        else:
            qubits_map = {
                q: cirq.LineQubit(x) for x, q in enumerate(circuit.all_qubits())
            }

    # Apply gates
    for gate in tqdm(circuit, disable=not verbose):

        # Check if qubits are missing
        if not gate.provides('qubits') or gate.qubits is None:
            raise ValueError(
                f"Gate(name='{gate.name}') requires {gate.n_qubits} qubits.")

        # Check if params are missing
        if gate.provides('params') and gate.params is None:
            raise ValueError(
                f"Gate(name='{gate.name}') requires {gate.n_params} parameters."
            )

        # Get mapped qubits
        qubits = [qubits_map[q] for q in gate.qubits]

        # Get params
        params = gate.params if gate.provides('params') else None

        if gate.name in {
                'MATRIX', 'U3', 'R_PI_2'
        } or (gate.provides('is_conjugated') and
              gate.is_conjugated()) or (gate.provides('is_transposed') and
                                        gate.is_transposed()):
            cirq_gate = cirq.MatrixGate(gate.matrix())
        elif gate.name[:5] == 'SQRT_':
            cirq_gate = _to_cirq_naming[gate.name[5:]](params)**(0.5 *
                                                                 gate.power)
        elif gate.name in _to_cirq_naming:
            cirq_gate = _to_cirq_naming[gate.name](params)**gate.power
        else:
            raise ValueError(f"{gate} not yet supported.")

        # Append
        circ.append(cirq_gate.on(*qubits))

    return circ
