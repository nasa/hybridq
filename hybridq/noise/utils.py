"""
Authors: Salvatore Mandra (salvatore.mandra@nasa.gov),
         Jeffrey Marshall (jeffrey.s.marshall@nasa.gov)

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
from hybridq.dm.circuit import Circuit as SuperCircuit
from hybridq.noise.channel.channel import BaseChannel
from hybridq.noise.channel import channel
from tqdm.auto import tqdm


def add_depolarizing_noise(circuit: Circuit,
                           probs: {float, list[float, ...], dict[any, float]},
                           where: {'before', 'after'} = 'after',
                           verbose: bool = False):
    """
    Given a `Circuit`, add global depolarizing noise after each instance
    of a `Gate`, with the same locality as the gate.
    Note, noise will not be added after an instance of `BaseChannel`

    circuit: Circuit
        The `Circuit` which will be modified. Note, a new `Circuit` is
        returned (this is not in place).
    probs: {float, list[float, ...], dict[any, float]}
        Depolarizing probabilities for `circuit`. If `probs` is a single `float`,
        the same probability is applied to all gates regardless the number of
        qubits. If `probs` is a list, the k-th value is used as the probability
        for all the k-qubit gates. If `probs` is a `dict`, `probs[k]` will be
        used as probability for k-qubit gates. If `probs[k]` is missing, the
        probability for a k-qubit gate will fallback to `probs[any]` (if
        provided).
    where: {'before', 'after', 'both'}
        Add noise either `'before'` or `'after'` every gate (default: `after`).
    verbose: bool, optional
        Verbose output.
    """
    from hybridq.circuit import Circuit

    # Check 'where'
    if where not in ['before', 'after']:
        raise ValueError("'where' can only be either 'before' or 'after'")

    # Convert circuit
    circuit = Circuit(circuit)

    # Convert probs
    probs = channel.__get_params(keys=sorted(set(g.n_qubits for g in circuit)),
                                 args=probs,
                                 value_type=float)

    # Define how to add noise
    def _add_noise(g):
        # Get probability
        p = probs[g.n_qubits]

        # Get noise
        noise = channel.GlobalDepolarizingChannel(g.qubits, p)

        # Return gates
        return [g] if isinstance(g, BaseChannel) else (
            [g, noise] if where == 'after' else [noise, g])

    # Update circuit
    return SuperCircuit(g for w in tqdm(
        circuit, disable=not verbose, desc='Add depolarizing noise')
                        for g in _add_noise(w))


def add_dephasing_noise(circuit: Circuit,
                        probs: {float, list[float, ...], dict[any, float]},
                        pauli_indexes: {int, list[int, ...], dict[any,
                                                                  int]} = 3,
                        where: {'before', 'after'} = 'after',
                        verbose: bool = False):
    """
    Given a `Circuit`, add dephasing noise after each instance of a `Gate`,
    which acts independently on the qubits of the gate.
    Note, noise will not be added after an instance of `BaseChannel`

    circuit: Circuit
        The `Circuit` which will be modified. Note, a new `Circuit` is
        returned (this is not in place).
    probs: {float, list[float, ...], dict[any, float]}
        Dephasing probabilities for `circuit`. If `probs` is a single `float`,
        the same probability is applied to all qubits. If `probs` is a list,
        the k-th value is used as the probability for the k-th qubit. If `probs`
        is a `dict`, `probs[q]` will be used as probability for the qubit `q`.
        If `probs[q]` is missing, the probability for a qubit `q` will fallback
        to `probs[any]` (if provided).
    pauli_indexes: {int, list[int, ...], dict[any, int]}
        Pauli indexes representing the dephasing axis (Pauli matrix).  If
        `pauli_indexes` is a single `int`, the same axis is applied to all
        qubits. If `pauli_indexes` is a list, the k-th value is used as the
        axis for the k-th qubit. If `pauli_indexes` is a `dict`,
        `pauli_indexes[q]` will be used as axis for the qubit `q`. If
        `pauli_indexes[q]` is missing, the axis for a qubit `q` will fallback
        to `pauli_indexes[any]` (if provided).
    where: {'before', 'after', 'both'}
        Add noise either `'before'` or `'after'` every gate (default: `after`).
    verbose: bool, optional
        Verbose output.
    """
    from hybridq.circuit import Circuit

    # Check 'where'
    if where not in ['before', 'after']:
        raise ValueError("'where' can only be either 'before' or 'after'")

    # Convert circuit
    circuit = Circuit(circuit)

    # Get all qubits
    qubits = circuit.all_qubits()

    # Convert gammas and probs
    probs = channel.__get_params(qubits, probs, value_type=float)
    pauli_indexes = channel.__get_params(qubits, pauli_indexes, value_type=int)

    # Define how to add noise
    def _add_noise(g):
        # Get gammas and probs
        _probs = {q: probs[q] for q in g.qubits}
        _axis = {q: pauli_indexes[q] for q in g.qubits}

        # Get noise
        noise = channel.LocalDephasingChannel(g.qubits,
                                              p=_probs,
                                              pauli_index=_axis)

        # Return gates
        return [g] if isinstance(
            g, BaseChannel) else ((g,) + noise if where == 'after' else noise +
                                  (g,))

    # Update circuit
    return SuperCircuit(g for w in tqdm(
        circuit, disable=not verbose, desc='Add amplitude damping noise')
                        for g in _add_noise(w))


def add_amplitude_damping_noise(
        circuit: Circuit,
        gammas: {float, list[float, ...], dict[any, float]},
        probs: {float, list[float, ...], dict[any, float]} = 1,
        where: {'before', 'after'} = 'after',
        verbose: bool = False):
    """
    Given a `Circuit`, add amplitude damping noise after each instance of a
     `Gate`. The noise will act independently on the qubits in the gate.
    Note, noise will not be added after an instance of `BaseChannel`

    circuit: Circuit
        The `Circuit` which will be modified. Note, a new `Circuit` is
        returned (this is not in place).
    gammas: {float, list[float, ...], dict[any, float]}
        Transition rate (0->1 and 1->0) for the amplitude damping noise
        channel. If `gammas` is a single `float`, the same probability is applied
        to all qubits. If `gammas` is a list, the k-th value is used as the
        probability for the k-th qubit. If `gammas` is a `dict`, `gammas[q]` will
        be used as probability for the qubit `q`. If `gammas[q]` is missing,
        the probability for a qubit `q` will fallback to `gammas[any]` (if
        provided).
    probs: {float, list[float, ...], dict[any, float]}
        Amplitude damping probabilities for `circuit`. If `probs` is a single
        `float`, the same probability is applied to all qubits. If `probs` is a
        list, the k-th value is used as the probability for the k-th qubit. If
        `probs` is a `dict`, `probs[q]` will be used as probability for the qubit
        `q`. If `probs[q]` is missing, the probability for a qubit `q` will
        fallback to `probs[any]` (if provided).
    where: {'before', 'after', 'both'}
        Add noise either `'before'` or `'after'` every gate (default: `after`).
    verbose: bool, optional
        Verbose output.
    """
    from hybridq.circuit import Circuit

    # Check 'where'
    if where not in ['before', 'after']:
        raise ValueError("'where' can only be either 'before' or 'after'")

    # Convert circuit
    circuit = Circuit(circuit)

    # Get all qubits
    qubits = circuit.all_qubits()

    # Convert gammas and probs
    gammas = channel.__get_params(qubits, gammas, value_type=float)
    probs = channel.__get_params(qubits, probs, value_type=float)

    # Define how to add noise
    def _add_noise(g):
        # Get gammas and probs
        _gammas = {q: gammas[q] for q in g.qubits}
        _probs = {q: probs[q] for q in g.qubits}

        # Get noise
        noise = channel.AmplitudeDampingChannel(g.qubits,
                                                gamma=_gammas,
                                                p=_probs)

        # Return gates
        return [g] if isinstance(
            g, BaseChannel) else ((g,) + noise if where == 'after' else noise +
                                  (g,))

    # Update circuit
    return SuperCircuit(g for w in tqdm(
        circuit, disable=not verbose, desc='Add amplitude damping noise')
                        for g in _add_noise(w))
