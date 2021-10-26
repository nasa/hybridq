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
    Given a `Circuit`, add depolarizing noise after each instance of a `Gate`,
    with the same locality as the gate.  Note, noise will not be added after an
    instance of `BaseChannel`

    circuit: Circuit
        The `Circuit` which will be modified. Note, a new `Circuit` is
        returned (this is not in place).
    probs: {float, list[float, ...], dict[any, float]}
        Depolarizing probabilities for `circuit`. If probs is a single `float`,
        the same probability is applied to all gates regardless the number of
        qubits. If `probs` is a list, the k-th value is used a the probability
        for all the k-qubit gates. If probs is a `dict`, `probs[k]` will be
        used as probability for k-qubit gates. If `probs[k]` is missing, the
        probability for a k-qubit gate will fallback to `probs[any]` (if
        provided).
    where: {'before', 'after', 'both'}
        Add noise either `'before'` or `'after'` every gate (default: `after`).
    verbose: bool, optional
        Verbose output.
    """
    from hybridq.circuit import Circuit
    # Check where
    if where not in ['before', 'after']:
        raise ValueError("'where' can only be either 'before' or 'after'")

    # Convert circuit
    circuit = Circuit(circuit)

    # Get all qubits
    qubits = circuit.all_qubits()

    # Convert probs
    probs = __get_params(qubits, probs, value_type=float)

    # Define how to add noise
    def _add_noise(g):
        # Get probability
        if g.n_qubits in probs:
            p = probs[g.n_qubits]
        elif any in probs:
            p = probs[any]
        else:
            raise ValueError(f"Params for '{g.n_qubits}' qubits not provided")

        # Get noise
        noise = channel.GlobalDepolarizingChannel(g.qubits, p)

        # Return gates
        return [g] if isinstance(g, BaseChannel) else (
            [g, noise] if where == 'after' else [noise, g])

    # Update circuit
    return SuperCircuit(g for w in tqdm(
        circuit, disable=not verbose, desc='Add depolarizing noise')
                        for g in _add_noise(w))


def __get_params(keys,
                 args,
                 key_type: callable = lambda x: x,
                 value_type: callable = lambda x: x):
    from hybridq.utils import to_dict, to_list

    # Initialize output
    _args = None

    # Try to convert to float
    if _args is None:
        try:
            _args = {any: value_type(args)}
        except:
            pass

    # Try to convert to dict
    if _args is None:
        try:
            _args = to_dict(args, key_type=key_type, value_type=value_type)
        except:
            pass

    # Try to convert to list
    if _args is None:
        try:
            # Convert to list
            _args = to_list(args, value_type=value_type)

        except:
            pass

        else:
            # Check number of keys
            if len(_args) != len(keys):
                raise ValueError("Must have exactly one value per qubit")

            # Get dict
            _args = {key_type(k): x for k, x in zip(keys, _args)}

    # If _args is still None, raise
    if _args is None:
        raise TypeError(f"'{args}' not supported")

    # Check keys
    if any not in _args and set(keys) != set(_args):
        raise ValueError("All keys must be specified")

    # Return
    return _args
